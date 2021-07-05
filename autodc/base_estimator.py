import os
import sys
import traceback
import time
import numpy as np
import pandas as pd
import shutil

from autodc.automl import AutoML
from autodc.components.feature_engineering.transformation_graph import DataNode
from autodc.utils.logging_utils import get_logger

class BaseEstimator(object):
    def __init__(
            self,
            dataset_name='default_dataset_name',
            time_limit=300,
            amount_of_resource=None,
            metric='acc',
            include_algorithms=None,
            include_preprocessors=None,
            enable_meta_algorithm_selection=True,
            enable_fe=True,
            ensemble_method='ensemble_selection',
            ensemble_size=50,
            per_run_time_limit=150,
            random_state=1,
            n_jobs=1,
            evaluation='holdout',
            output_dir="/tmp/Soln_results",
            delete_output_dir_after_fit=False):
        self.dataset_name = dataset_name
        self.metric = metric
        self.task_type = None
        self.time_limit = time_limit
        self.amount_of_resource = amount_of_resource
        self.include_algorithms = include_algorithms
        self.include_preprocessors = include_preprocessors
        self.enable_meta_algorithm_selection = enable_meta_algorithm_selection
        self.enable_fe = enable_fe
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.per_run_time_limit = per_run_time_limit
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.evaluation = evaluation
        self._ml_engine = None
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)

        # Create output directory.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dir_postfix = "autodc-%s" % time.time()
        self.output_dir = os.path.join(output_dir, output_dir_postfix)
        # Create output directory.
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.delete_output_dir = delete_output_dir_after_fit

    def get_output_dir(self):
        return self.output_dir

    def build_engine(self):
        """Build AutoML controller"""
        engine = self.get_automl()(
            dataset_name=self.dataset_name,
            task_type=self.task_type,
            metric=self.metric,
            time_limit=self.time_limit,
            amount_of_resource=self.amount_of_resource,
            include_algorithms=self.include_algorithms,
            include_preprocessors=self.include_preprocessors,
            enable_meta_algorithm_selection=self.enable_meta_algorithm_selection,
            enable_fe=self.enable_fe,
            ensemble_method=self.ensemble_method,
            ensemble_size=self.ensemble_size,
            per_run_time_limit=self.per_run_time_limit,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            evaluation=self.evaluation,
            output_dir=self.output_dir
        )
        return engine

    def fit(self, data: DataNode, **kwargs):
        assert data is not None and isinstance(data, DataNode)
        self._ml_engine = self.build_engine()
        self._ml_engine.fit(data, **kwargs)
        if self.delete_output_dir:
            shutil.rmtree(self.output_dir)
        return self

    def predict(self, X: DataNode, batch_size=None, n_jobs=1):
        return self._ml_engine.predict(X)

    def score(self, data: DataNode):
        return self._ml_engine.score(data)

    def refit(self):
        return self._ml_engine.refit()

    def predict_proba(self, X: DataNode, batch_size=None, n_jobs=1):
        return self._ml_engine.predict_proba(X)

    def get_automl(self):
        return AutoML

    def show_info(self):
        raise NotImplementedError()

    @property
    def best_hpo_config(self):
        return self._ml_engine.solver.best_hpo_config

    @property
    def best_algo_id(self):
        return self._ml_engine.solver.optimal_algo_id

    @property
    def nbest_algo_id(self):
        return self._ml_engine.solver.nbest_algo_ids

    @property
    def best_perf(self):
        return self._ml_engine.solver.incumbent_perf

    @property
    def best_node(self):
        return self._ml_engine.solver.best_data_node

    @property
    def best_fe_config(self):
        return self._ml_engine.solver.best_data_node.config

    def data_transform(self, data: DataNode):
        return self._ml_engine.solver.fe_optimizer.apply(data, self._ml_engine.solver.best_data_node)

    def feature_corelation(self, data: DataNode):
        X0, y0 = data.data
        X, y = self.data_transformer(data).data
        i = X0.shape[1]
        j = X.shape[1]
        corre_mat = np.zeros([i, j])
        for it in range(i):
            for jt in range(j):
                corre_mat[it, jt] = np.corrcoef(X0[:, it], X[:, jt])[0, 1]
        df = pd.DataFrame(corre_mat)
        df.columns = ['origin_fearure' + str(it) for it in range(i)]
        df.index = ['transformed_fearure' + str(jt) for jt in range(j)]
        return df

    def feature_origin(self):
        conf = self._ml_engine.solver.best_data_node.config
        pro_table = []
        for process in ['preprocessor1', 'preprocessor2', 'balancer', 'rescaler', 'generator', 'selector']:
            if (conf[process] == 'empty'):
                pro_hash = {'Processor': process, 'Algorithm': None, 'File_path': None, 'Arguments': None}
                pro_table.append(pro_hash)
                continue

            pro_hash = {'Processor': process, 'Algorithm': conf[process]}
            argstr = ''
            for key in conf:
                if (key.find(conf[process]) != -1):
                    arg = key.replace(conf[process] + ':', '')
                    argstr += (arg + '=' + str(conf[key]) + '  ')
            pro_hash['Arguments'] = argstr
            pathstr = './autodc/components/feature_engineering/transformations/'
            if (process == 'preprocessor1'):
                pro_hash['File_path'] = pathstr + 'continous_discretizer.py'
                pro_table.append(pro_hash)
                continue

            if (process == 'preprocessor2'):
                pro_hash['File_path'] = pathstr + 'discrete_categorizer.py'
                pro_table.append(pro_hash)
                continue

            if (process == 'balancer'):
                pro_hash['File_path'] = pathstr + 'preprocessor/' + conf[process] + '.py'
                pro_table.append(pro_hash)
                continue

            pro_hash['File_path'] = pathstr + process + '/' + conf[process] + '.py'
            pro_table.append(pro_hash)

        df = pd.DataFrame(pro_table)[['Processor', 'Algorithm', 'File_path', 'Arguments']]
        df.index = ['step' + str(i) for i in range(1, 7)]
        return df

    def get_val_stats(self):
        return self._ml_engine.get_val_stats()

    def get_ens_model_info(self):
        return self._ml_engine.get_ens_model_info()

    def get_ens_model_details(self):
        return self._ml_engine.get_ens_model_details()
