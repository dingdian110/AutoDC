from ConfigSpace import ConfigurationSpace, UnParametrizedHyperparameter, CategoricalHyperparameter
import warnings
import os
import time
import numpy as np
import pickle as pkl
from multiprocessing import Lock
from sklearn.metrics.scorer import balanced_accuracy_scorer, _ThresholdScorer
from sklearn.preprocessing import OneHotEncoder

from autodc.utils.logging_utils import get_logger
from autodc.components.evaluators.base_evaluator import _BaseEvaluator
from autodc.components.evaluators.evaluate_func import validation
from autodc.components.fe_optimizers.ano_bo_optimizer import get_task_hyperparameter_space
from autodc.components.fe_optimizers.parse import parse_config, construct_node
from autodc.components.evaluators.base_evaluator import CombinedTopKModelSaver
from autodc.components.models.classification import _classifiers, _addons
from autodc.components.utils.constants import *


def get_estimator(config, estimator_id):
    classifier_type = estimator_id
    config_ = config.copy()
    # config_.pop('estimator', None)
    config_['%s:random_state' % classifier_type] = 1
    hpo_config = dict()
    for key in config_:
        key_name = key.split(':')[0]
        if classifier_type == key_name:
            act_key = key.split(':')[1]
            hpo_config[act_key] = config_[key]
    try:
        estimator = _classifiers[classifier_type](**hpo_config)
    except:
        estimator = _addons.components[classifier_type](**hpo_config)
    if hasattr(estimator, 'n_jobs'):
        setattr(estimator, 'n_jobs', 1)
    return classifier_type, estimator


def get_hpo_cs(estimator_id, task_type=CLASSIFICATION):
    if estimator_id in _classifiers:
        clf_class = _classifiers[estimator_id]
    elif estimator_id in _addons.components:
        clf_class = _addons.components[estimator_id]
    else:
        raise ValueError("Algorithm %s not supported!" % estimator_id)
    cs = clf_class.get_hyperparameter_search_space()
    return cs


def get_fe_cs(estimator_id, task_type=0, include_preprocessors=None):
    cs = get_task_hyperparameter_space(task_type=task_type, estimator_id=estimator_id,
                                       include_preprocessors=include_preprocessors)
    return cs


def get_combined_cs(estimator_id, task_type=0, include_preprocessors=None):
    cs = ConfigurationSpace()
    hpo_cs = get_hpo_cs(estimator_id, task_type)
    fe_cs = get_fe_cs(estimator_id, task_type, include_preprocessors=include_preprocessors)
    config_cand = [estimator_id]
    config_option = CategoricalHyperparameter('hpo', config_cand)
    cs.add_hyperparameter(config_option)
    for config_item in config_cand:
        sub_configuration_space = hpo_cs
        parent_hyperparameter = {'parent': config_option,
                                 'value': config_item}
        cs.add_configuration_space(config_item, sub_configuration_space,
                                   parent_hyperparameter=parent_hyperparameter)
    for hp in fe_cs.get_hyperparameters():
        cs.add_hyperparameter(hp)
    for cond in fe_cs.get_conditions():
        cs.add_condition(cond)
    for bid in fe_cs.get_forbiddens():
        cs.add_forbidden_clause(bid)
    # model = UnParametrizedHyperparameter("estimator", estimator_id)
    # cs.add_hyperparameter(model)
    return cs


class CombinedClassificationEvaluator(_BaseEvaluator):
    def __init__(self, estimator_id, scorer=None, data_node=None, task_type=0, resampling_strategy='cv',
                 resampling_params=None, timestamp=None, output_dir=None, seed=1):
        self.resampling_strategy = resampling_strategy
        self.resampling_params = resampling_params

        self.estimator_id = estimator_id
        self.scorer = scorer if scorer is not None else balanced_accuracy_scorer
        self.task_type = task_type
        self.data_node = data_node
        self.output_dir = output_dir
        self.seed = seed
        self.onehot_encoder = None
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        self.continue_training = False

        self.train_node = data_node.copy_()
        self.val_node = data_node.copy_()

        self.timestamp = timestamp
        # TODO: Top-k k?
        self.topk_model_saver = CombinedTopKModelSaver(k=60, model_dir=self.output_dir, identifier=timestamp)

    def get_fit_params(self, y, estimator):
        from autodc.components.utils.balancing import get_weights
        _init_params, _fit_params = get_weights(
            y, estimator, None, {}, {})
        return _init_params, _fit_params

    def __call__(self, config, **kwargs):
        start_time = time.time()
        return_dict = dict()
        self.seed = 1
        downsample_ratio = kwargs.get('resource_ratio', 1.0)

        if 'holdout' in self.resampling_strategy:
            try:
                # Prepare data node.
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")

                    if self.resampling_params is None or 'test_size' not in self.resampling_params:
                        test_size = 0.33
                    else:
                        test_size = self.resampling_params['test_size']

                    from sklearn.model_selection import StratifiedShuffleSplit
                    ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=self.seed)
                    for train_index, test_index in ss.split(self.data_node.data[0], self.data_node.data[1]):
                        _x_train, _x_val = self.data_node.data[0][train_index], self.data_node.data[0][test_index]
                        _y_train, _y_val = self.data_node.data[1][train_index], self.data_node.data[1][test_index]
                    self.train_node.data = [_x_train, _y_train]
                    self.val_node.data = [_x_val, _y_val]

                    data_node, op_list = parse_config(self.train_node, config, record=True)
                    _val_node = self.val_node.copy_()
                    _val_node = construct_node(_val_node, op_list)

                _x_train, _y_train = data_node.data
                _x_val, _y_val = _val_node.data

                config_dict = config.get_dictionary().copy()
                # Prepare training and initial params for classifier.
                init_params, fit_params = {}, {}
                if data_node.enable_balance == 1:
                    init_params, fit_params = self.get_fit_params(_y_train, self.estimator_id)
                    for key, val in init_params.items():
                        config_dict[key] = val

                if data_node.data_balance == 1:
                    fit_params['data_balance'] = True

                classifier_id, clf = get_estimator(config_dict, self.estimator_id)

                if self.onehot_encoder is None:
                    self.onehot_encoder = OneHotEncoder(categories='auto')
                    y = np.reshape(_y_train, (len(_y_train), 1))
                    self.onehot_encoder.fit(y)

                score = validation(clf, self.scorer, _x_train, _y_train, _x_val, _y_val,
                                   random_state=self.seed,
                                   onehot=self.onehot_encoder if isinstance(self.scorer,
                                                                            _ThresholdScorer) else None,
                                   fit_params=fit_params)

                if 'rw_lock' not in kwargs or kwargs['rw_lock'] is None:
                    self.logger.info('rw_lock not defined! Possible read-write conflicts may happen!')
                lock = kwargs.get('rw_lock', Lock())
                lock.acquire()
                if np.isfinite(score):
                    save_flag, model_path, delete_flag, model_path_deleted = self.topk_model_saver.add(config, score,
                                                                                                       classifier_id)
                    if save_flag is True:
                        with open(model_path, 'wb') as f:
                            pkl.dump([op_list, clf], f)
                        self.logger.info("Model saved to %s" % model_path)

                    try:
                        if delete_flag and os.path.exists(model_path_deleted):
                            os.remove(model_path_deleted)
                            self.logger.info("Model deleted from %s" % model_path_deleted)
                    except:
                        pass
                lock.release()

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.logger.info('Evaluator: %s' % (str(e)))
                score = -np.inf

        elif 'cv' in self.resampling_strategy:
            # Prepare data node.
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")

                    if 'cv' in self.resampling_strategy:
                        if self.resampling_params is None or 'folds' not in self.resampling_params:
                            folds = 5
                        else:
                            folds = self.resampling_params['folds']

                    from sklearn.model_selection import StratifiedKFold
                    skfold = StratifiedKFold(n_splits=folds, random_state=self.seed, shuffle=False)
                    scores = list()

                    for train_index, test_index in skfold.split(self.data_node.data[0], self.data_node.data[1]):
                        _x_train, _x_val = self.data_node.data[0][train_index], self.data_node.data[0][test_index]
                        _y_train, _y_val = self.data_node.data[1][train_index], self.data_node.data[1][test_index]
                        self.train_node.data = [_x_train, _y_train]
                        self.val_node.data = [_x_val, _y_val]

                        data_node, op_list = parse_config(self.train_node, config, record=True)
                        _val_node = self.val_node.copy_()
                        _val_node = construct_node(_val_node, op_list)

                        _x_train, _y_train = data_node.data
                        _x_val, _y_val = _val_node.data

                        config_dict = config.get_dictionary().copy()
                        # Prepare training and initial params for classifier.
                        init_params, fit_params = {}, {}
                        if data_node.enable_balance == 1:
                            init_params, fit_params = self.get_fit_params(_y_train, self.estimator_id)
                            for key, val in init_params.items():
                                config_dict[key] = val

                        if data_node.data_balance == 1:
                            fit_params['data_balance'] = True

                        classifier_id, clf = get_estimator(config_dict, self.estimator_id)

                        if self.onehot_encoder is None:
                            self.onehot_encoder = OneHotEncoder(categories='auto')
                            y = np.reshape(_y_train, (len(_y_train), 1))
                            self.onehot_encoder.fit(y)

                        _score = validation(clf, self.scorer, _x_train, _y_train, _x_val, _y_val,
                                            random_state=self.seed,
                                            onehot=self.onehot_encoder if isinstance(self.scorer,
                                                                                     _ThresholdScorer) else None,
                                            fit_params=fit_params)
                        scores.append(_score)
                    score = np.mean(scores)

                # TODO: Don't save models for cv
                if 'rw_lock' not in kwargs or kwargs['rw_lock'] is None:
                    self.logger.info('rw_lock not defined! Possible read-write conflicts may happen!')
                lock = kwargs.get('rw_lock', Lock())
                lock.acquire()
                if np.isfinite(score):
                    _ = self.topk_model_saver.add(config, score, classifier_id)
                lock.release()

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.logger.info('Evaluator: %s' % (str(e)))
                score = -np.inf

        elif 'partial' in self.resampling_strategy:
            try:
                # Prepare data node.
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")

                    if self.resampling_params is None or 'test_size' not in self.resampling_params:
                        test_size = 0.33
                    else:
                        test_size = self.resampling_params['test_size']

                    from sklearn.model_selection import StratifiedShuffleSplit
                    ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=self.seed)
                    for train_index, test_index in ss.split(self.data_node.data[0], self.data_node.data[1]):
                        _x_train, _x_val = self.data_node.data[0][train_index], self.data_node.data[0][test_index]
                        _y_train, _y_val = self.data_node.data[1][train_index], self.data_node.data[1][test_index]
                    self.train_node.data = [_x_train, _y_train]
                    self.val_node.data = [_x_val, _y_val]

                    data_node, op_list = parse_config(self.train_node, config, record=True)
                    _val_node = self.val_node.copy_()
                    _val_node = construct_node(_val_node, op_list)

                _x_train, _y_train = data_node.data

                if downsample_ratio != 1:
                    down_ss = StratifiedShuffleSplit(n_splits=1, test_size=downsample_ratio,
                                                     random_state=self.seed)
                    for _, _val_index in down_ss.split(_x_train, _y_train):
                        _act_x_train, _act_y_train = _x_train[_val_index], _y_train[_val_index]
                else:
                    _act_x_train, _act_y_train = _x_train, _y_train
                    _val_index = list(range(len(_x_train)))

                _x_val, _y_val = _val_node.data

                config_dict = config.get_dictionary().copy()
                # Prepare training and initial params for classifier.
                init_params, fit_params = {}, {}
                if data_node.enable_balance == 1:
                    init_params, fit_params = self.get_fit_params(_y_train, self.estimator_id)
                    for key, val in init_params.items():
                        config_dict[key] = val
                if 'sample_weight' in fit_params:
                    fit_params['sample_weight'] = fit_params['sample_weight'][_val_index]
                if data_node.data_balance == 1:
                    fit_params['data_balance'] = True

                classifier_id, clf = get_estimator(config_dict, self.estimator_id)

                if self.onehot_encoder is None:
                    self.onehot_encoder = OneHotEncoder(categories='auto')
                    y = np.reshape(_y_train, (len(_y_train), 1))
                    self.onehot_encoder.fit(y)
                score = validation(clf, self.scorer, _act_x_train, _act_y_train, _x_val, _y_val,
                                   random_state=self.seed,
                                   onehot=self.onehot_encoder if isinstance(self.scorer,
                                                                            _ThresholdScorer) else None,
                                   fit_params=fit_params)

                # TODO: Only save models with maximum resources
                if 'rw_lock' not in kwargs or kwargs['rw_lock'] is None:
                    self.logger.info('rw_lock not defined! Possible read-write conflicts may happen!')
                lock = kwargs.get('rw_lock', Lock())
                lock.acquire()
                if np.isfinite(score) and downsample_ratio == 1:
                    save_flag, model_path, delete_flag, model_path_deleted = self.topk_model_saver.add(config, score,
                                                                                                       classifier_id)
                    if save_flag is True:
                        with open(model_path, 'wb') as f:
                            pkl.dump([op_list, clf], f)
                        self.logger.info("Model saved to %s" % model_path)

                    try:
                        if delete_flag and os.path.exists(model_path_deleted):
                            os.remove(model_path_deleted)
                            self.logger.info("Model deleted from %s" % model_path_deleted)
                    except:
                        pass
                lock.release()
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.logger.info('Evaluator: %s' % (str(e)))
                score = -np.inf

        else:
            raise ValueError('Invalid resampling strategy: %s!' % self.resampling_strategy)

        try:
            self.logger.info('Evaluation<%s> | Score: %.4f | Time cost: %.2f seconds | Shape: %s' %
                             (classifier_id,
                              self.scorer._sign * score,
                              time.time() - start_time, _x_train.shape))
        except:
            pass

        # Turn it into a minimization problem.
        return_dict['score'] = -score
        return -score
