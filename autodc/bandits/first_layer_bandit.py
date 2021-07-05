import os
import time
import numpy as np
import pickle as pkl
from typing import List
from sklearn.metrics import accuracy_score
from autodc.components.metrics.metric import get_metric
from autodc.utils.constant import MAX_INT
from autodc.components.feature_engineering.transformation_graph import DataNode
from autodc.bandits.second_layer_bandit import SecondLayerBandit
from autodc.components.evaluators.base_evaluator import load_transformer_estimator, load_combined_transformer_estimator
from autodc.components.fe_optimizers.parse import construct_node
from autodc.utils.logging_utils import get_logger
from autodc.components.utils.constants import CLS_TASKS


class FirstLayerBandit(object):
    def __init__(self, task_type, trial_num,
                 classifier_ids: List[str], data: DataNode,
                 include_preprocessors=None,
                 time_limit=None,
                 metric='acc',
                 ensemble_method='ensemble_selection',
                 ensemble_size=50,
                 per_run_time_limit=300,
                 output_dir="logs",
                 dataset_name='default_dataset',
                 eval_type='holdout',
                 inner_opt_algorithm='fixed',
                 enable_fe=True,
                 fe_algo='bo',
                 n_jobs=1,
                 seed=1):
        """
        :param classifier_ids: subset of {'adaboost','bernoulli_nb','decision_tree','extra_trees','gaussian_nb','gradient_boosting',
        'gradient_boosting','k_nearest_neighbors','lda','liblinear_svc','libsvm_svc','multinomial_nb','passive_aggressive','qda',
        'random_forest','sgd'}
        """
        self.timestamp = time.time()
        self.task_type = task_type
        self.include_preprocessors = include_preprocessors
        self.metric = get_metric(metric)
        self.original_data = data.copy_()
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.trial_num = trial_num
        self.n_jobs = n_jobs
        self.alpha = 4
        self.seed = seed
        self.output_dir = output_dir
        self.early_stop_flag = False
        # np.random.seed(self.seed)

        # Best configuration.
        self.optimal_algo_id = None
        self.nbest_algo_ids = None
        self.best_lower_bounds = None
        self.es = None

        # Set up backend.
        self.dataset_name = dataset_name
        self.time_limit = time_limit
        self.start_time = time.time()
        self.logger = get_logger('AutoDC: %s' % dataset_name)

        # Bandit settings.
        self.incumbent_perf = -float("INF")
        self.arms = classifier_ids
        self.include_algorithms = classifier_ids
        self.rewards = dict()
        self.sub_bandits = dict()
        self.evaluation_cost = dict()
        self.eval_type = eval_type
        self.enable_fe = enable_fe
        self.fe_algo = fe_algo
        self.inner_opt_algorithm = inner_opt_algorithm

        # Record the execution cost for each arm.
        if not (self.time_limit is None) ^ (self.trial_num is None):
            raise ValueError('Please set one of time_limit or trial_num.')

        self.arm_cost_stats = dict()
        for _arm in self.arms:
            self.arm_cost_stats[_arm] = list()

        for arm in self.arms:
            self.rewards[arm] = list()
            self.evaluation_cost[arm] = list()
            self.sub_bandits[arm] = SecondLayerBandit(
                self.task_type, arm, self.original_data,
                include_preprocessors=self.include_preprocessors,
                metric=self.metric,
                output_dir=output_dir,
                per_run_time_limit=per_run_time_limit,
                seed=self.seed,
                eval_type=eval_type,
                dataset_id=dataset_name,
                n_jobs=self.n_jobs,
                fe_algo=fe_algo,
                mth=self.inner_opt_algorithm,
                timestamp=self.timestamp
            )

        self.action_sequence = list()
        self.final_rewards = list()
        self.start_time = time.time()
        self.time_records = list()

    def get_stats(self):
        return self.time_records, self.final_rewards

    def get_eval_dict(self):
        # Call after fit
        eval_dict = {}
        for _arm in self.arms:
            if self.inner_opt_algorithm == 'combined':
                eval_dict.update(self.sub_bandits[_arm].eval_dict)
            else:
                fe_eval_dict = self.sub_bandits[_arm].eval_dict['fe']
                hpo_eval_dict = self.sub_bandits[_arm].eval_dict['hpo']
                eval_dict.update(fe_eval_dict)
                eval_dict.update(hpo_eval_dict)
        return eval_dict

    def optimize(self):
        if self.inner_opt_algorithm in ['rb_hpo', 'fixed', 'alter_hpo', 'alter', 'combined']:
            self.optimize_explore_first()
        elif self.inner_opt_algorithm == 'equal':
            self.optimize_equal_resource()
        else:
            raise ValueError('Unsupported optimization method: %s!' % self.inner_opt_algorithm)

        scores = list()
        for _arm in self.arms:
            scores.append(self.sub_bandits[_arm].incumbent_perf)
        scores = np.array(scores)
        algo_idx = np.argmax(scores)
        self.optimal_algo_id = self.arms[algo_idx]
        self.incumbent_perf = scores[algo_idx]
        _threshold, _ensemble_size = self.incumbent_perf * 0.90, 5
        if self.incumbent_perf < 0.:
            _threshold = self.incumbent_perf / 0.9

        idxs = np.argsort(-scores)[:_ensemble_size]
        _algo_ids = [self.arms[idx] for idx in idxs]
        self.nbest_algo_ids = list()
        for _idx, _arm in zip(idxs, _algo_ids):
            if scores[_idx] >= _threshold:
                self.nbest_algo_ids.append(_arm)
        assert len(self.nbest_algo_ids) > 0

        self.logger.info('=' * 50)
        self.logger.info('Best_algo_perf:  %s' % str(self.incumbent_perf))
        self.logger.info('Best_algo_id:    %s' % str(self.optimal_algo_id))
        self.logger.info('Nbest_algo_ids:  %s' % str(self.nbest_algo_ids))
        self.logger.info('Arm candidates:  %s' % str(self.arms))
        self.logger.info('Best val scores: %s' % str(list(scores)))
        self.logger.info('=' * 50)

        if self.inner_opt_algorithm == 'combined':
            self.best_config = self.sub_bandits[self.optimal_algo_id].incumbent_config
            if self.best_config is None:
                raise ValueError("The best configuration is None! Check if the evaluator fails or try larger budget!")
        else:
            # Fit the best model
            eval_dict = dict()
            fe_eval_dict = self.sub_bandits[self.optimal_algo_id].eval_dict['fe']
            hpo_eval_dict = self.sub_bandits[self.optimal_algo_id].eval_dict['hpo']
            eval_dict.update(fe_eval_dict)
            eval_dict.update(hpo_eval_dict)
            self.best_fe_config, self.best_hpo_config = \
                sorted(eval_dict.items(), key=lambda t: t[1][0], reverse=True)[0][0]

        if self.ensemble_method is not None:
            if self.inner_opt_algorithm == 'combined':
                config_path = os.path.join(self.output_dir, '%s_topk_config.pkl' % self.timestamp)
                with open(config_path, 'rb') as f:
                    stats = pkl.load(f)

                from autodc.components.ensemble.combined_ensemble.ensemble_bulider import EnsembleBuilder
            else:
                config_path = os.path.join(self.output_dir, '%s_topk_config.pkl' % self.timestamp)
                with open(config_path, 'rb') as f:
                    stats = pkl.load(f)

                from autodc.components.ensemble import EnsembleBuilder

            # Ensembling all intermediate/ultimate models found in above optimization process.
            self.es = EnsembleBuilder(stats=stats,
                                      data_node=self.original_data,
                                      ensemble_method=self.ensemble_method,
                                      ensemble_size=self.ensemble_size,
                                      task_type=self.task_type,
                                      metric=self.metric,
                                      output_dir=self.output_dir)
            self.es.fit(data=self.original_data)

    def refit(self):
        if self.ensemble_method is not None:
            self.es.refit()

    def _predict(self, test_data: DataNode):
        if self.ensemble_method is not None:
            if self.es is None:
                raise AttributeError("AutoML is not fitted!")
            return self.es.predict(test_data)
        else:
            if self.inner_opt_algorithm == 'combined':
                best_op_list, estimator = load_combined_transformer_estimator(self.output_dir, self.best_config,
                                                                              self.timestamp)
            else:
                best_op_list, estimator = load_transformer_estimator(self.output_dir, self.optimal_algo_id,
                                                                     self.best_hpo_config, self.best_fe_config,
                                                                     self.timestamp)

            test_data_node = test_data.copy_()
            test_data_node = construct_node(test_data_node, best_op_list)

            if self.task_type in CLS_TASKS:
                return estimator.predict_proba(test_data_node.data[0])
            else:
                return estimator.predict(test_data_node.data[0])

    def predict_proba(self, test_data: DataNode):
        if self.task_type not in CLS_TASKS:
            raise AttributeError("predict_proba is not supported in regression")
        return self._predict(test_data)

    def predict(self, test_data: DataNode):
        if self.task_type in CLS_TASKS:
            pred = self._predict(test_data)
            return np.argmax(pred, axis=-1)
        else:
            return self._predict(test_data)

    def score(self, test_data: DataNode, metric_func=None):
        if metric_func is None:
            self.logger.info('Metric is set to accuracy_score by default!')
            metric_func = accuracy_score
        y_pred = self.predict(test_data)
        return metric_func(test_data.data[1], y_pred)

    def optimize_explore_first(self):
        # Initialize the parameters.
        arm_num = len(self.arms)
        arm_candidate = self.arms.copy()
        self.best_lower_bounds = np.zeros(arm_num)
        _iter_id = 0
        if self.time_limit is None:
            if arm_num * self.alpha > self.trial_num:
                raise ValueError('Trial number should be larger than %d.' % (arm_num * self.alpha))
        else:
            self.trial_num = MAX_INT

        while _iter_id < self.trial_num:
            if _iter_id < arm_num * self.alpha:
                _arm = self.arms[_iter_id % arm_num]
                self.logger.info('Optimize %s in the %d-th iteration' % (_arm, _iter_id))
                _start_time = time.time()
                reward = self.sub_bandits[_arm].play_once(self.time_limit - _start_time + self.start_time)
                self.arm_cost_stats[_arm].append(time.time() - _start_time)

                self.rewards[_arm].append(reward)
                self.action_sequence.append(_arm)
                self.final_rewards.append(reward)
                self.time_records.append(time.time() - self.start_time)
                if reward > self.incumbent_perf:
                    self.incumbent_perf = reward
                    self.optimal_algo_id = _arm
                self.logger.info('The best performance found for %s is %.4f' % (_arm, reward))
                _iter_id += 1
            else:
                # Pull each arm in the candidate once.
                for _arm in arm_candidate:
                    self.logger.info('Optimize %s in the %d-th iteration' % (_arm, _iter_id))
                    _start_time = time.time()
                    reward = self.sub_bandits[_arm].play_once(self.time_limit - _start_time + self.start_time)
                    self.arm_cost_stats[_arm].append(time.time() - _start_time)

                    self.rewards[_arm].append(reward)
                    self.action_sequence.append(_arm)
                    self.final_rewards.append(reward)
                    self.time_records.append(time.time() - self.start_time)

                    self.logger.info('The best performance found for %s is %.4f' % (_arm, reward))
                    _iter_id += 1

            if _iter_id >= arm_num * self.alpha:
                # Update the upper/lower bound estimation.
                budget_left = max(self.time_limit - (time.time() - self.start_time), 0)
                avg_cost = np.array([np.mean(self.arm_cost_stats[_arm]) for _arm in arm_candidate]).mean()
                steps = int(budget_left / avg_cost)
                upper_bounds, lower_bounds = list(), list()

                for _arm in arm_candidate:
                    rewards = self.rewards[_arm]
                    slope = (rewards[-1] - rewards[-self.alpha]) / self.alpha
                    if self.time_limit is None:
                        steps = self.trial_num - _iter_id
                    upper_bound = np.min([1.0, rewards[-1] + slope * steps])
                    upper_bounds.append(upper_bound)
                    lower_bounds.append(rewards[-1])
                    self.best_lower_bounds[self.arms.index(_arm)] = rewards[-1]

                # Reject the sub-optimal arms.
                n = len(arm_candidate)
                flags = [False] * n
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            if upper_bounds[i] < lower_bounds[j]:
                                flags[i] = True

                if np.sum(flags) == n:
                    self.logger.error('Removing all the arms simultaneously!')
                self.logger.info('Candidates  : %s' % ','.join(arm_candidate))
                self.logger.info('Upper bound : %s' % ','.join(['%.4f' % val for val in upper_bounds]))
                self.logger.info('Lower bound : %s' % ','.join(['%.4f' % val for val in lower_bounds]))
                self.logger.info('Arms removed: %s' % [item for idx, item in enumerate(arm_candidate) if flags[idx]])

                # Update the arm_candidates.
                arm_candidate = [item for index, item in enumerate(arm_candidate) if not flags[index]]

            self.early_stop_flag = True
            for arm in arm_candidate:
                if not self.sub_bandits[arm].early_stopped_flag:
                    self.early_stop_flag = False

            if self.time_limit is not None and time.time() > self.start_time + self.time_limit:
                break
            if self.early_stop_flag:
                self.logger.info("Maximum configuration number met for each arm candidate!")
                break
        return self.final_rewards

    def optimize_equal_resource(self):
        arm_num = len(self.arms)
        arm_candidate = self.arms.copy()
        if self.time_limit is None:
            resource_per_algo = self.trial_num // arm_num
        else:
            resource_per_algo = 8
        for _arm in arm_candidate:
            self.sub_bandits[_arm].total_resource = resource_per_algo
            self.sub_bandits[_arm].mth = 'fixed'

        _iter_id = 0
        while _iter_id < self.trial_num:
            _arm = self.arms[_iter_id % arm_num]
            self.logger.info('Optimize %s in the %d-th iteration' % (_arm, _iter_id))
            reward = self.sub_bandits[_arm].play_once()

            self.rewards[_arm].append(reward)
            self.action_sequence.append(_arm)
            self.final_rewards.append(reward)
            self.time_records.append(time.time() - self.start_time)
            if reward > self.incumbent_perf:
                self.incumbent_perf = reward
                self.optimal_algo_id = _arm
                self.logger.info('The best performance found for %s is %.4f' % (_arm, reward))
            _iter_id += 1

            if self.time_limit is not None and time.time() > self.start_time + self.time_limit:
                break
        return self.final_rewards

    # def __del__(self):
    #     for _arm in self.arms:
    #         del self.sub_bandits[_arm].optimizer
