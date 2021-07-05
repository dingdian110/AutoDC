import time
import os
import numpy as np
from autodc.components.hpo_optimizer.base_optimizer import BaseHPOptimizer, MAX_INT
from autodc.components.hpo_optimizer.base.mfsebase import MfseBase


class MfseOptimizer(BaseHPOptimizer, MfseBase):
    def __init__(self, evaluator, config_space, time_limit=None, evaluation_limit=None,
                 per_run_time_limit=600, per_run_mem_limit=1024, output_dir='./', inner_iter_num_per_iter=1, seed=1,
                 R=27, eta=3, n_jobs=1):
        BaseHPOptimizer.__init__(self, evaluator, config_space, seed)
        MfseBase.__init__(self, eval_func=self.evaluator, config_space=self.config_space,
                          seed=seed, R=R, eta=eta, n_jobs=n_jobs)
        self.time_limit = time_limit
        self.evaluation_num_limit = evaluation_limit
        self.inner_iter_num_per_iter = inner_iter_num_per_iter
        self.per_run_time_limit = per_run_time_limit
        self.per_run_mem_limit = per_run_mem_limit

    def iterate(self, budget=MAX_INT):
        '''
            Iterate a SH procedure (inner loop) in Hyperband.
        :return:
        '''
        _start_time = time.time()
        for _ in range(self.inner_iter_num_per_iter):
            _time_elapsed = time.time() - _start_time
            if _time_elapsed >= budget:
                break
            budget_left = budget - _time_elapsed
            self._iterate(self.s_values[self.inner_iter_id], budget=budget_left)
            self.inner_iter_id = (self.inner_iter_id + 1) % (self.s_max + 1)

            # Remove tmp model
            if self.evaluator.continue_training:
                for filename in os.listdir(self.evaluator.model_dir):
                    # Temporary model
                    if 'tmp_%s' % self.evaluator.timestamp in filename:
                        try:
                            filepath = os.path.join(self.evaluator.model_dir, filename)
                            os.remove(filepath)
                        except Exception:
                            pass

        if len(self.incumbent_perfs) > 0:
            inc_idx = np.argmin(np.array(self.incumbent_perfs))

            for idx in range(len(self.incumbent_perfs)):
                if hasattr(self.evaluator, 'fe_config'):
                    fe_config = self.evaluator.fe_config
                else:
                    fe_config = None
                self.eval_dict[(fe_config, self.incumbent_configs[idx])] = [-self.incumbent_perfs[idx], time.time()]

            self.incumbent_perf = -self.incumbent_perfs[inc_idx]
            self.incumbent_config = self.incumbent_configs[inc_idx]

        self.perfs = self.incumbent_perfs
        self.configs = self.incumbent_configs

        # Incumbent performance: the large, the better.
        iteration_cost = time.time() - _start_time
        return self.incumbent_perf, iteration_cost, self.incumbent_config

    def get_evaluation_stats(self):
        return self.evaluation_stats
