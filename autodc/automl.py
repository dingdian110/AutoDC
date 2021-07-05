import os
import sys
import traceback
from autodc.utils.logging_utils import setup_logger, get_logger
from autodc.components.metrics.metric import get_metric
from autodc.components.utils.constants import CLS_TASKS, RGS_TASKS
from autodc.components.ensemble import ensemble_list
from autodc.components.feature_engineering.transformation_graph import DataNode
from autodc.components.models.regression import _regressors
from autodc.components.models.classification import _classifiers
from autodc.components.models.imbalanced_classification import _imb_classifiers
from autodc.components.meta_learning.algorithm_recomendation.ranknet_advisor import RankNetAdvisor
from autodc.bandits.first_layer_bandit import FirstLayerBandit

classification_algorithms = _classifiers.keys()
imb_classication_algorithms = _imb_classifiers.keys()
regression_algorithms = _regressors.keys()


class AutoML(object):
    def __init__(self, time_limit=300,
                 dataset_name='default_name',
                 amount_of_resource=None,
                 task_type=None,
                 metric='bal_acc',
                 include_algorithms=None,
                 include_preprocessors=None,
                 ensemble_method='ensemble_selection',
                 enable_meta_algorithm_selection=True,
                 enable_fe=True,
                 per_run_time_limit=150,
                 ensemble_size=50,
                 evaluation='holdout',
                 output_dir="logs",
                 logging_config=None,
                 random_state=1,
                 n_jobs=1):
        self.metric_id = metric
        self.metric = get_metric(self.metric_id)

        self.dataset_name = dataset_name
        self.time_limit = time_limit
        self.seed = random_state
        self.per_run_time_limit = per_run_time_limit
        self.output_dir = output_dir
        self.logging_config = logging_config
        self.logger = self._get_logger(self.dataset_name)

        self.evaluation_type = evaluation
        self.include_preprocessors = include_preprocessors

        self.amount_of_resource = amount_of_resource
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.enable_meta_algorithm_selection = enable_meta_algorithm_selection
        self.enable_fe = enable_fe
        self.task_type = task_type
        self.n_jobs = n_jobs
        self.solver = None

        # Disable meta learning
        if self.include_preprocessors is not None:
            self.enable_meta_algorithm_selection = False

        if include_algorithms is not None:
            self.include_algorithms = include_algorithms
        else:
            if task_type in CLS_TASKS:
                self.include_algorithms = list(classification_algorithms)
            elif task_type in RGS_TASKS:
                self.include_algorithms = list(regression_algorithms)
            else:
                raise ValueError("Unknown task type %s" % task_type)
        if ensemble_method is not None and ensemble_method not in ensemble_list:
            raise ValueError("%s is not supported for ensemble!" % ensemble_method)

    def _get_logger(self, name):
        logger_name = 'AutoDC-%s(%d)' % (name, self.seed)
        setup_logger(os.path.join(self.output_dir, '%s.log' % str(logger_name)),
                     self.logging_config,
                     )
        return get_logger(logger_name)

    def fit(self, train_data: DataNode, **kwargs):
        """
        This function includes this following two procedures.
            1. tune each algorithm's hyperparameters.
            2. engineer each algorithm's features automatically.
        :param train_data:
        :return:
        """
        # Check whether this dataset is balanced or not.
        # if self.task_type in CLS_TASKS and is_unbalanced_dataset(train_data):
        #     self.logger.info('Input dataset is imbalanced!')
        #     train_data = DataBalancer().operate(train_data)

        dataset_id = kwargs.get('dataset_id', None)
        inner_opt_algorithm = kwargs.get('opt_strategy', 'alter_hpo')
        self.logger.info('Optimization algorithm in 2rd bandit: %s' % inner_opt_algorithm)

        if self.enable_meta_algorithm_selection:
            try:
                n_algo_recommended = 5
                meta_datasets = kwargs.get('meta_datasets', None)
                self.logger.info('Executing Meta-Learning based Algorithm Recommendation.')
                include_models = ['extra_trees', 'adaboost', 'liblinear_svc', 'random_forest',
                                  'libsvm_svc', 'lightgbm']
                self.include_algorithms = include_models
                self.logger.info('Final Algorithms Recommended: [%s]' % ','.join(self.include_algorithms))
            except Exception as e:
                self.logger.error("Meta-Learning based Algorithm Recommendation FAILED: %s." % str(e))
                traceback.print_exc(file=sys.stdout)

        self.solver = FirstLayerBandit(self.task_type, self.amount_of_resource,
                                       self.include_algorithms, train_data,
                                       include_preprocessors=self.include_preprocessors,
                                       per_run_time_limit=self.per_run_time_limit,
                                       dataset_name=self.dataset_name,
                                       ensemble_method=self.ensemble_method,
                                       ensemble_size=self.ensemble_size,
                                       inner_opt_algorithm=inner_opt_algorithm,
                                       metric=self.metric,
                                       enable_fe=self.enable_fe,
                                       fe_algo='bo',
                                       seed=self.seed,
                                       time_limit=self.time_limit,
                                       eval_type=self.evaluation_type,
                                       output_dir=self.output_dir)
        self.solver.optimize()

    def refit(self):
        self.solver.refit()

    def predict_proba(self, test_data: DataNode):
        return self.solver.predict_proba(test_data)

    def predict(self, test_data: DataNode):
        return self.solver.predict(test_data)

    def score(self, test_data: DataNode, metric_func=None):
        if metric_func is None:
            metric_func = self.metric
        return metric_func(self, test_data, test_data.data[1])

    def get_ens_model_info(self):
        if self.ensemble_method is not None:
            return self.solver.es.get_ens_model_info()
        else:
            return None

    def get_val_stats(self):
        return self.solver.get_stats()

    def get_ens_model_details(self):
        if self.ensemble_method is not None:
            return self.solver.es.get_ens_model_details()
        else:
            return None
