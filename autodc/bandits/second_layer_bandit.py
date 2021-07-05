import copy
import time
import os
import numpy as np
import pickle as pkl
from autodc.components.evaluators.cls_evaluator import ClassificationEvaluator
from autodc.components.evaluators.rgs_evaluator import RegressionEvaluator
from autodc.utils.logging_utils import get_logger
from autodc.components.feature_engineering.transformation_graph import DataNode
from autodc.components.fe_optimizers import build_fe_optimizer
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
from autodc.components.fe_optimizers.ano_bo_optimizer import get_task_hyperparameter_space
from autodc.components.hpo_optimizer import build_hpo_optimizer
from autodc.components.utils.constants import CLS_TASKS, RGS_TASKS
from autodc.utils.decorators import time_limit
from autodc.utils.constant import MAX_INT


class SecondLayerBandit(object):
    def __init__(self, task_type, estimator_id: str, data: DataNode, metric,
                 include_preprocessors=None,
                 share_fe=False, output_dir='logs',
                 per_run_time_limit=120,
                 per_run_mem_limit=5120,
                 dataset_id='default',
                 eval_type='holdout',
                 mth='rb', sw_size=3,
                 n_jobs=1, seed=1,
                 enable_fe=True, fe_algo='bo',
                 number_of_unit_resource=2,
                 total_resource=30,
                 timestamp=None):
        self.task_type = task_type
        self.metric = metric
        self.number_of_unit_resource = number_of_unit_resource
        # One unit of resource, that's, the number of trials per iteration.
        self.one_unit_of_resource = 5
        self.total_resource = total_resource
        self.per_run_time_limit = per_run_time_limit
        self.per_run_mem_limit = per_run_mem_limit
        self.estimator_id = estimator_id
        self.include_preprocessors = include_preprocessors
        self.evaluation_type = eval_type
        self.original_data = data.copy_()
        self.share_fe = share_fe
        self.output_dir = output_dir
        self.n_jobs = n_jobs
        self.mth = mth
        self.seed = seed
        self.sliding_window_size = sw_size
        task_id = '%s-%d-%s' % (dataset_id, seed, estimator_id)
        self.logger = get_logger(self.__class__.__name__ + '-' + task_id)

        # Bandit settings.
        # self.arms = ['fe', 'hpo']
        self.arms = ['hpo', 'fe']
        self.rewards = dict()
        self.optimizer = dict()
        self.evaluation_cost = dict()
        self.update_flag = dict()
        # Global incumbent.
        self.inc = dict()
        self.local_inc = dict()
        self.local_hist = {'fe': [], 'hpo': []}
        self.inc_record = {'fe': list(), 'hpo': list()}
        self.exp_output = dict()
        self.eval_dict = {'fe': dict(), 'hpo': dict()}
        for arm in self.arms:
            self.rewards[arm] = list()
            self.update_flag[arm] = False
            self.evaluation_cost[arm] = list()
            self.exp_output[arm] = dict()
        self.pull_cnt = 0
        self.action_sequence = list()
        self.final_rewards = list()
        self.incumbent_config = None
        self.incumbent_perf = float("-INF")
        self.early_stopped_flag = False
        self.first_start = True

        # Fetch hyperparameter space.
        if self.task_type in CLS_TASKS:
            from autodc.components.models.classification import _classifiers, _addons
            if estimator_id in _classifiers:
                clf_class = _classifiers[estimator_id]
            elif estimator_id in _addons.components:
                clf_class = _addons.components[estimator_id]
            else:
                raise ValueError("Algorithm %s not supported!" % estimator_id)
            cs = clf_class.get_hyperparameter_search_space()
            # model = UnParametrizedHyperparameter("estimator", estimator_id)
            # cs.add_hyperparameter(model)
        elif self.task_type in RGS_TASKS:
            from autodc.components.models.regression import _regressors, _addons
            if estimator_id in _regressors:
                reg_class = _regressors[estimator_id]
            elif estimator_id in _addons.components:
                reg_class = _addons.components[estimator_id]
            else:
                raise ValueError("Algorithm %s not supported!" % estimator_id)
            cs = reg_class.get_hyperparameter_search_space()
            # model = UnParametrizedHyperparameter("estimator", estimator_id)
            # cs.add_hyperparameter(model)
        else:
            raise ValueError("Unknown task type %s!" % self.task_type)

        self.config_space = cs
        self.default_config = cs.get_default_configuration()
        self.config_space.seed(self.seed)

        self.fe_config_space = get_task_hyperparameter_space(self.task_type,
                                                             self.estimator_id,
                                                             include_preprocessors=self.include_preprocessors)
        self.fe_default_config = self.fe_config_space.get_default_configuration()

        self.timestamp = timestamp
        # Build the Feature Engineering component.
        if self.task_type in CLS_TASKS:
            fe_evaluator = ClassificationEvaluator(self.default_config, self.fe_default_config, estimator_id, scorer=self.metric,
                                                   data_node=self.original_data, name='fe',
                                                   resampling_strategy=self.evaluation_type,
                                                   seed=self.seed, output_dir=self.output_dir,
                                                   timestamp=self.timestamp)
            hpo_evaluator = ClassificationEvaluator(self.default_config, self.fe_default_config, estimator_id, scorer=self.metric,
                                                    data_node=self.original_data, name='hpo',
                                                    resampling_strategy=self.evaluation_type,
                                                    seed=self.seed, output_dir=self.output_dir,
                                                    timestamp=self.timestamp)

        elif self.task_type in RGS_TASKS:
            fe_evaluator = RegressionEvaluator(self.default_config, self.fe_default_config, estimator_id, scorer=self.metric,
                                               data_node=self.original_data, name='fe',
                                               resampling_strategy=self.evaluation_type,
                                               seed=self.seed, output_dir=self.output_dir,
                                               timestamp=self.timestamp)
            hpo_evaluator = RegressionEvaluator(self.default_config, self.fe_default_config, estimator_id, scorer=self.metric,
                                                data_node=self.original_data, name='hpo',
                                                resampling_strategy=self.evaluation_type,
                                                seed=self.seed, output_dir=self.output_dir,
                                                timestamp=self.timestamp)
        else:
            raise ValueError('Invalid task type!')

        if self.mth != 'combined':
            self.enable_fe = enable_fe
            self.fe_algo = fe_algo
            self.optimizer['fe'] = build_fe_optimizer(self.fe_algo, self.evaluation_type,
                                                      self.task_type, self.fe_config_space,
                                                      self.original_data, fe_evaluator,
                                                      estimator_id, per_run_time_limit,
                                                      per_run_mem_limit, self.seed,
                                                      shared_mode=self.share_fe, n_jobs=n_jobs)

            self.inc['fe'], self.local_inc['fe'] = self.fe_default_config, self.fe_default_config

            # Build the HPO component.
            # trials_per_iter = max(len(self.optimizer['fe'].trans_types), 20)
            trials_per_iter = self.one_unit_of_resource * self.number_of_unit_resource

            self.optimizer['hpo'] = build_hpo_optimizer(self.evaluation_type, hpo_evaluator, cs, output_dir=output_dir,
                                                        per_run_time_limit=per_run_time_limit,
                                                        inner_iter_num_per_iter=trials_per_iter,
                                                        seed=self.seed, n_jobs=n_jobs)

            self.inc['hpo'], self.local_inc['hpo'] = self.default_config, self.default_config
            self.init_config = cs.get_default_configuration()
            self.local_hist['fe'].append(self.fe_default_config)
            self.local_hist['hpo'].append(self.default_config)

        else:
            self.rewards = list()
            self.evaluation_cost = list()
            self.eval_dict = {}
            trials_per_iter = self.one_unit_of_resource * self.number_of_unit_resource

            if self.task_type in CLS_TASKS:
                from autodc.utils.combined_cls_evaluator import get_combined_cs
                from autodc.utils.combined_cls_evaluator import CombinedClassificationEvaluator as CombinedEvaluator
            else:
                from autodc.utils.combined_rgs_evaluator import get_combined_cs
                from autodc.utils.combined_rgs_evaluator import CombinedRegressionEvaluator as CombinedEvaluator

            self.evaluator = CombinedEvaluator(
                estimator_id,
                scorer=self.metric,
                data_node=self.original_data,
                timestamp=self.timestamp,
                output_dir=self.output_dir,
                resampling_strategy=self.evaluation_type)
            cs = get_combined_cs(self.estimator_id, self.task_type, include_preprocessors=self.include_preprocessors)

            self.optimizer = build_hpo_optimizer(self.evaluation_type, self.evaluator, cs,
                                                 output_dir=self.output_dir,
                                                 per_run_time_limit=self.per_run_time_limit,
                                                 inner_iter_num_per_iter=10,
                                                 seed=self.seed, n_jobs=self.n_jobs)

    def collect_iter_stats(self, _arm, results):
        pre_inc_perf = self.incumbent_perf
        for arm_id in self.arms:
            self.update_flag[arm_id] = False

        if _arm == 'fe' and len(self.final_rewards) == 0:
            self.incumbent_perf = self.optimizer['fe'].baseline_score
            self.final_rewards.append(self.incumbent_perf)

        self.logger.debug('After %d-th pulling, results: %s' % (self.pull_cnt, results))

        self.eval_dict[_arm].update(self.optimizer[_arm].eval_dict)

        score, iter_cost, config = results
        if score is None:
            score = 0.0
        self.rewards[_arm].append(score)
        self.evaluation_cost[_arm].append(iter_cost)
        self.local_inc[_arm] = config

        if self.evaluation_type == 'partial':
            self.exp_output[_arm].update(self.optimizer[_arm].exp_output)

        # Update global incumbent from FE and HPO.
        if np.isfinite(score) and score > self.incumbent_perf:
            self.inc[_arm] = config
            self.local_hist[_arm].append(config)
            if _arm == 'fe':
                if self.mth not in ['alter_hpo', 'rb_hpo', 'fixed']:
                    self.inc['hpo'] = self.default_config
                else:
                    self.inc['hpo'] = self.init_config
            else:
                if self.mth not in ['alter_hpo', 'rb_hpo', 'fixed']:
                    self.inc['fe'] = self.original_data

            self.incumbent_perf = score

            arm_id = 'fe' if _arm == 'hpo' else 'hpo'
            self.update_flag[arm_id] = True

            if self.mth in ['rb_hpo', 'alter_hpo'] and _arm == 'fe':
                self.prepare_optimizer(arm_id)

            if self.mth in ['rb_hpo', 'alter_hpo'] and _arm == 'hpo':
                if len(self.rewards[_arm]) == 1:
                    self.prepare_optimizer(arm_id)
                    self.init_config = config
                    if config != self.default_config:
                        self.logger.info('Initial hp_config for FE has changed!')

            if self.mth in ['alter_p', 'fixed']:
                self.prepare_optimizer(arm_id)

        post_inc_perf = self.incumbent_perf
        if np.isfinite(pre_inc_perf) and np.isfinite(post_inc_perf):
            self.inc_record[_arm].append(post_inc_perf - pre_inc_perf)
        else:
            self.inc_record[_arm].append(0.)

    def optimize_rb(self, remaining_budget=MAX_INT):
        # First pull each arm #sliding_window_size times.
        if self.pull_cnt < len(self.arms) * self.sliding_window_size:
            arm_picked = self.arms[self.pull_cnt % 2]
        else:
            imp_values = list()
            for _arm in self.arms:
                # increasing_rewards = get_increasing_sequence(self.rewards[_arm])
                # impv = list()
                # for idx in range(1, len(increasing_rewards)):
                #     impv.append(increasing_rewards[idx] - increasing_rewards[idx - 1])
                # imp_values.append(np.mean(impv[-self.sliding_window_size:]))
                imp_values.append(np.mean(self.inc_record[_arm][-self.sliding_window_size:]))
                print(self.inc_record[_arm])

            self.logger.debug('Imp values: %s' % imp_values)
            if imp_values[0] == imp_values[1]:
                # Break ties randomly.
                # arm_picked = np.random.choice(self.arms, 1)[0]
                arm_picked = 'fe' if self.action_sequence[-1] == 'hpo' else 'hpo'
            else:
                arm_picked = self.arms[np.argmax(imp_values)]

        # Early stopping scenario.
        if self.optimizer[arm_picked].early_stopped_flag is True:
            arm_picked = 'hpo' if arm_picked == 'fe' else 'fe'
            if self.optimizer[arm_picked].early_stopped_flag is True:
                self.early_stopped_flag = True
                return

        self.action_sequence.append(arm_picked)
        self.logger.info(','.join(self.action_sequence))
        self.logger.info('Pulling arm: %s for %s at %d-th round' % (arm_picked, self.estimator_id, self.pull_cnt))
        results = self.optimizer[arm_picked].iterate(budget=remaining_budget)
        self.collect_iter_stats(arm_picked, results)
        self.pull_cnt += 1

    def optimize_alternatedly(self, remaining_budget=MAX_INT):
        # First choose one arm.
        _arm = self.arms[self.pull_cnt % 2]
        self.logger.debug('Pulling arm: %s for %s at %d-th round' % (_arm, self.estimator_id, self.pull_cnt))
        if self.first_start is True and _arm == 'hpo':
            if hasattr(self.optimizer[_arm], 'init_hpo_iter_num'):
                # self.optimizer[_arm].init_hpo_iter_num = 20
                self.optimizer[_arm].init_hpo_iter_num = 10
            else:
                raise ValueError('Optimizer does not have property - init_hpo_iter_num.')
            self.first_start = False

        # Early stopping scenario.
        if self.optimizer[_arm].early_stopped_flag is True:
            arm_picked = 'hpo' if _arm == 'fe' else 'fe'
            if self.optimizer[arm_picked].early_stopped_flag is True:
                self.early_stopped_flag = True
                return

        # Execute one iteration.
        results = self.optimizer[_arm].iterate(budget=remaining_budget)
        self.collect_iter_stats(_arm, results)
        self.action_sequence.append(_arm)
        self.pull_cnt += 1

    def optimize_combined(self, remaining_budget=MAX_INT):
        score, iter_cost, config = self.optimizer.iterate(budget=remaining_budget)
        self.eval_dict.update(self.optimizer.eval_dict)
        self.rewards.append(score)
        if max(self.rewards) == score and not np.isinf(max(self.rewards)):
            self.incumbent_perf = score
            self.incumbent_config = config
        self.evaluation_cost.append(iter_cost)

        # Early stopping scenario.
        if self.optimizer.early_stopped_flag is True:
            self.early_stopped_flag = True
            return

    def optimize_fixed_pipeline(self, remaining_budget=MAX_INT):
        if self.pull_cnt <= 3:
            _arm = 'hpo'
        else:
            _arm = 'fe'
        self.logger.debug('Pulling arm: %s for %s at %d-th round' % (_arm, self.estimator_id, self.pull_cnt))

        # Execute one iteration.
        results = self.optimizer[_arm].iterate(budget=remaining_budget)
        self.collect_iter_stats(_arm, results)
        self.action_sequence.append(_arm)
        self.pull_cnt += 1

    def optimize_one_component(self, mth, remaining_budget=MAX_INT):
        _arm = 'hpo' if mth == 'hpo_only' else 'fe'
        self.logger.debug('Pulling arm: %s for %s at %d-th round' % (_arm, self.estimator_id, self.pull_cnt))

        if self.optimizer[_arm].early_stopped_flag is True:
            self.early_stopped_flag = True
            return

        # Execute one iteration.
        results = self.optimizer[_arm].iterate(budget=remaining_budget)
        self.collect_iter_stats(_arm, results)
        self.action_sequence.append(_arm)
        self.pull_cnt += 1

    def evaluate_joint_solution(self):
        # Update join incumbent from FE and HPO.
        _perf = None
        try:
            with time_limit(self.per_run_time_limit):
                if self.task_type in CLS_TASKS:
                    evaluator = ClassificationEvaluator(
                        self.local_inc['hpo'], self.local_inc['fe'], self.estimator_id,
                        data_node=self.original_data, scorer=self.metric,
                        name='hpo', resampling_strategy=self.evaluation_type,
                        seed=self.seed, output_dir=self.output_dir, timestamp=self.timestamp)
                else:
                    evaluator = RegressionEvaluator(
                        self.local_inc['hpo'], self.local_inc['fe'], self.estimator_id,
                        data_node=self.original_data, scorer=self.metric,
                        name='hpo', resampling_strategy=self.evaluation_type,
                        seed=self.seed, output_dir=self.output_dir, timestamp=self.timestamp)
                _perf = -evaluator(self.local_inc['hpo'])
        except Exception as e:
            self.logger.error(str(e))
        # Update INC.
        if _perf is not None and np.isfinite(_perf) and _perf > self.incumbent_perf:
            self.inc['hpo'] = self.local_inc['hpo']
            self.inc['fe'] = self.local_inc['fe']
            self.incumbent_perf = _perf

    def play_once(self, remaining_budget=MAX_INT):
        start_time = time.time()
        if self.early_stopped_flag:
            self.logger.warning("Already explored 70 percent of the hp space or maximum configuration number met: %d" %
                                self.optimizer.maximum_config_num)
            return self.incumbent_perf

        if self.mth in ['rb', 'rb_hpo']:
            self.optimize_rb(remaining_budget=remaining_budget)
            if time.time() - start_time < remaining_budget:
                self.evaluate_joint_solution()
        elif self.mth in ['alter', 'alter_p', 'alter_hpo']:
            self.optimize_alternatedly(remaining_budget=remaining_budget)
            if time.time() - start_time < remaining_budget:
                self.evaluate_joint_solution()
        elif self.mth in ['fe_only', 'hpo_only']:
            self.optimize_one_component(self.mth, remaining_budget=remaining_budget)
        elif self.mth in ['combined']:
            self.optimize_combined(remaining_budget=remaining_budget)
        elif self.mth in ['fixed']:
            self.optimize_fixed_pipeline(remaining_budget=remaining_budget)
        else:
            raise ValueError('Invalid method: %s' % self.mth)

        self.final_rewards.append(self.incumbent_perf)
        return self.incumbent_perf

    def prepare_optimizer(self, _arm):
        trials_per_iter = self.one_unit_of_resource * self.number_of_unit_resource
        if _arm == 'fe':
            # Build the Feature Engineering component.
            self.original_data._node_id = -1
            inc_hpo = copy.deepcopy(self.inc['hpo'])
            if self.task_type in CLS_TASKS:
                fe_evaluator = ClassificationEvaluator(inc_hpo, self.fe_default_config, self.estimator_id,
                                                       data_node=self.original_data, scorer=self.metric,
                                                       name='fe', resampling_strategy=self.evaluation_type,
                                                       seed=self.seed, output_dir=self.output_dir,
                                                       timestamp=self.timestamp)
            elif self.task_type in RGS_TASKS:
                fe_evaluator = RegressionEvaluator(inc_hpo, self.fe_default_config, self.estimator_id,
                                                   data_node=self.original_data, scorer=self.metric,
                                                   name='fe', resampling_strategy=self.evaluation_type,
                                                   seed=self.seed, output_dir=self.output_dir,
                                                   timestamp=self.timestamp)
            else:
                raise ValueError('Invalid task type!')
            self.optimizer[_arm] = build_fe_optimizer(self.fe_algo, self.evaluation_type,
                                                      self.task_type, self.fe_config_space,
                                                      self.original_data.copy_(),
                                                      fe_evaluator, self.estimator_id, self.per_run_time_limit,
                                                      self.per_run_mem_limit, self.seed,
                                                      shared_mode=self.share_fe,
                                                      number_of_unit_resource=self.number_of_unit_resource,
                                                      n_jobs=self.n_jobs)
        else:
            # trials_per_iter = self.optimizer['fe'].evaluation_num_last_iteration // 2
            # trials_per_iter = max(20, trials_per_iter)
            inc_fe = copy.deepcopy(self.inc['fe'])
            if self.task_type in CLS_TASKS:
                hpo_evaluator = ClassificationEvaluator(self.default_config, inc_fe, self.estimator_id, scorer=self.metric,
                                                        data_node=self.original_data, name='hpo',
                                                        resampling_strategy=self.evaluation_type,
                                                        seed=self.seed, output_dir=self.output_dir,
                                                        timestamp=self.timestamp)
            elif self.task_type in RGS_TASKS:
                hpo_evaluator = RegressionEvaluator(self.default_config, inc_fe, self.estimator_id, scorer=self.metric,
                                                    data_node=self.original_data, name='hpo',
                                                    resampling_strategy=self.evaluation_type,
                                                    seed=self.seed, output_dir=self.output_dir,
                                                    timestamp=self.timestamp)
            else:
                raise ValueError('Invalid task type!')

            self.optimizer[_arm] = build_hpo_optimizer(self.evaluation_type, hpo_evaluator, self.config_space,
                                                       output_dir=self.output_dir,
                                                       per_run_time_limit=self.per_run_time_limit,
                                                       inner_iter_num_per_iter=trials_per_iter, seed=self.seed)

        self.logger.debug('=' * 30)
        self.logger.debug('UPDATE OPTIMIZER: %s' % _arm)
        self.logger.debug('=' * 30)
