from sklearn.metrics.scorer import _BaseScorer
import numpy as np
import os
import pickle as pkl
import pandas as pd

from autodc.components.utils.constants import CLS_TASKS
from autodc.components.evaluators.base_evaluator import fetch_predict_estimator
from autodc.components.fe_optimizers.parse import construct_node
from autodc.components.ensemble.base_ensemble import BaseEnsembleModel
from functools import reduce


class Bagging(BaseEnsembleModel):
    def __init__(self, stats, data_node,
                 ensemble_size: int,
                 task_type: int,
                 metric: _BaseScorer,
                 output_dir=None):
        super().__init__(stats=stats,
                         data_node=data_node,
                         ensemble_method='bagging',
                         ensemble_size=ensemble_size,
                         task_type=task_type,
                         metric=metric,
                         output_dir=output_dir)

    def fit(self, datanode):
        return self

    def predict(self, data):
        model_pred_list = []
        final_pred = []
        # Get predictions from each model
        model_cnt = 0
        for algo_id in self.stats:
            model_to_eval = self.stats[algo_id]
            for idx, (_, _, path) in enumerate(model_to_eval):
                with open(path, 'rb')as f:
                    op_list, model = pkl.load(f)
                _node = data.copy_()

                _node = construct_node(_node, op_list)

                if self.base_model_mask[model_cnt] == 1:
                    if self.task_type in CLS_TASKS:
                        model_pred_list.append(model.predict_proba(_node.data[0]))
                    else:
                        model_pred_list.append(model.predict(_node.data[0]))
                model_cnt += 1

        # Calculate the average of predictions
        for i in range(len(data.data[0])):
            sample_pred_list = [model_pred[i] for model_pred in model_pred_list]
            pred_average = reduce(lambda x, y: x + y, sample_pred_list) / len(sample_pred_list)
            final_pred.append(pred_average)

        return np.array(final_pred)

    def get_ens_model_info(self):
        model_cnt = 0
        ens_info = {}
        ens_config = []
        for algo_id in self.stats:
            model_to_eval = self.stats[algo_id]
            for idx, (config, _, path) in enumerate(model_to_eval):
                if not hasattr(self, 'base_model_mask') or self.base_model_mask[model_cnt] == 1:
                    ens_config.append((algo_id, config, path, "\n"))
                model_cnt += 1
        ens_info['ensemble_method'] = 'bagging'
        ens_info['config'] = ens_config
        return ens_info

    def get_ens_model_details(self):
        model_cnt = 0
        ens_info = {}
        ens_config = []
        output_info = []
        for algo_id in self.stats:
            model_to_eval = self.stats[algo_id]
            for idx, (config, _, path) in enumerate(model_to_eval):
                if not hasattr(self, 'base_model_mask') or self.base_model_mask[model_cnt] == 1:
                    ens_config.append((algo_id, config, path))
                model_cnt += 1
        ens_info['ensemble_method'] = 'bagging'
        ens_info['config'] = ens_config

        for ids, (algo_id, config, model_path) in enumerate(ens_info['config']):
            output_info.append([algo_id, 1, config, model_path])
        ens_info_mat = pd.DataFrame(output_info, columns=['Algorithm', 'Weight', 'Configuration', 'Model_path'])
        import os
        ens_info_save_dir = os.getcwd()
        if not os.path.exists(ens_info_save_dir):
            os.makedirs(ens_info_save_dir)
        import random
        ens_info_mat.to_csv(
            os.path.join(ens_info_save_dir + "/Ensemble_model_details_" + str(random.randint(1, 10000)) + ".tsv"),
            sep="\t")
        return 0
