import numpy as np
import warnings
import os
import pickle as pkl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.scorer import _BaseScorer

from autodc.components.ensemble.base_ensemble import BaseEnsembleModel
from autodc.components.utils.constants import CLS_TASKS
from autodc.components.evaluators.base_evaluator import fetch_predict_estimator
from autodc.components.fe_optimizers.parse import construct_node


class Blending(BaseEnsembleModel):
    def __init__(self, stats, data_node,
                 ensemble_size: int,
                 task_type: int,
                 metric: _BaseScorer,
                 output_dir=None,
                 meta_learner='lightgbm'):
        super().__init__(stats=stats,
                         data_node=data_node,
                         ensemble_method='blending',
                         ensemble_size=ensemble_size,
                         task_type=task_type,
                         metric=metric,
                         output_dir=output_dir)
        try:
            from lightgbm import LGBMClassifier
        except:
            warnings.warn("Lightgbm is not imported! Blending will use linear model instead!")
            meta_learner = 'linear'
        self.meta_method = meta_learner
        # We use Xgboost as default meta-learner
        if self.task_type in CLS_TASKS:
            if meta_learner == 'linear':
                from sklearn.linear_model.logistic import LogisticRegression
                self.meta_learner = LogisticRegression(max_iter=1000)
            elif meta_learner == 'gb':
                from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
                self.meta_learner = GradientBoostingClassifier(learning_rate=0.05, subsample=0.7, max_depth=4,
                                                               n_estimators=250)
            elif meta_learner == 'lightgbm':
                from lightgbm import LGBMClassifier
                self.meta_learner = LGBMClassifier(max_depth=4, learning_rate=0.05, n_estimators=150)
        else:
            if meta_learner == 'linear':
                from sklearn.linear_model import LinearRegression
                self.meta_learner = LinearRegression()
            elif meta_learner == 'lightgbm':
                from lightgbm import LGBMRegressor
                self.meta_learner = LGBMRegressor(max_depth=4, learning_rate=0.05, n_estimators=70)

    def fit(self, data):
        # Split training data for phase 1 and phase 2
        test_size = 0.2

        # Train basic models using a part of training data
        model_cnt = 0
        suc_cnt = 0
        feature_p2 = None
        for algo_id in self.stats.keys():
            model_to_eval = self.stats[algo_id]
            for idx, (config, _, path) in enumerate(model_to_eval):
                with open(path, 'rb')as f:
                    op_list, model = pkl.load(f)
                _node = data.copy_()

                _node = construct_node(_node, op_list, mode='train')

                X, y = _node.data
                if self.task_type in CLS_TASKS:
                    x_p1, x_p2, y_p1, y_p2 = train_test_split(X, y, test_size=test_size,
                                                              stratify=data.data[1], random_state=1)
                else:
                    x_p1, x_p2, y_p1, y_p2 = train_test_split(X, y, test_size=test_size,
                                                              random_state=1)

                if self.base_model_mask[model_cnt] == 1:
                    estimator = fetch_predict_estimator(self.task_type, algo_id, config, x_p1, y_p1,
                                                        weight_balance=_node.enable_balance,
                                                        data_balance=_node.data_balance)
                    with open(os.path.join(self.output_dir, '%s-blending-model%d' % (self.timestamp, model_cnt)),
                              'wb') as f:
                        pkl.dump(estimator, f)
                    if self.task_type in CLS_TASKS:
                        pred = estimator.predict_proba(x_p2)
                        n_dim = np.array(pred).shape[1]
                        if n_dim == 2:
                            # Binary classificaion
                            n_dim = 1
                        # Initialize training matrix for phase 2
                        if feature_p2 is None:
                            num_samples = len(x_p2)
                            feature_p2 = np.zeros((num_samples, self.ensemble_size * n_dim))
                        if n_dim == 1:
                            feature_p2[:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] = pred[:, 1:2]
                        else:
                            feature_p2[:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] = pred
                    else:
                        pred = estimator.predict(x_p2).reshape(-1, 1)
                        n_dim = 1
                        # Initialize training matrix for phase 2
                        if feature_p2 is None:
                            num_samples = len(x_p2)
                            feature_p2 = np.zeros((num_samples, self.ensemble_size * n_dim))
                        feature_p2[:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] = pred
                    suc_cnt += 1
                model_cnt += 1
        self.meta_learner.fit(feature_p2, y_p2)

        return self

    def get_feature(self, data):
        # Predict the labels via blending
        feature_p2 = None
        model_cnt = 0
        suc_cnt = 0
        for algo_id in self.stats.keys():
            model_to_eval = self.stats[algo_id]
            for idx, (config, _, path) in enumerate(model_to_eval):
                with open(path, 'rb')as f:
                    op_list, model = pkl.load(f)
                _node = data.copy_()

                _node = construct_node(_node, op_list)

                if self.base_model_mask[model_cnt] == 1:
                    with open(os.path.join(self.output_dir, '%s-blending-model%d' % (self.timestamp, model_cnt)),
                              'rb') as f:
                        estimator = pkl.load(f)
                    if self.task_type in CLS_TASKS:
                        pred = estimator.predict_proba(_node.data[0])
                        n_dim = np.array(pred).shape[1]
                        if n_dim == 2:
                            # Binary classificaion
                            n_dim = 1
                        # Initialize training matrix for phase 2
                        if feature_p2 is None:
                            num_samples = len(data.data[0])
                            feature_p2 = np.zeros((num_samples, self.ensemble_size * n_dim))
                        if n_dim == 1:
                            feature_p2[:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] = pred[:, 1:2]
                        else:
                            feature_p2[:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] = pred
                    else:
                        pred = estimator.predict(_node.data[0]).reshape(-1, 1)
                        n_dim = 1
                        # Initialize training matrix for phase 2
                        if feature_p2 is None:
                            num_samples = len(data.data[0])
                            feature_p2 = np.zeros((num_samples, self.ensemble_size * n_dim))
                        feature_p2[:, suc_cnt * n_dim:(suc_cnt + 1) * n_dim] = pred
                    suc_cnt += 1
                model_cnt += 1

        return feature_p2

    def predict(self, data):
        feature_p2 = self.get_feature(data)
        # Get predictions from meta-learner
        if self.task_type in CLS_TASKS:
            final_pred = self.meta_learner.predict_proba(feature_p2)
        else:
            final_pred = self.meta_learner.predict(feature_p2)
        return final_pred

    def get_ens_model_info(self):
        model_cnt = 0
        ens_info = {}
        ens_config = []
        for algo_id in self.stats:
            model_to_eval = self.stats[algo_id]
            for idx, (config, _, _) in enumerate(model_to_eval):
                if not hasattr(self, 'base_model_mask') or self.base_model_mask[model_cnt] == 1:
                    model_path = os.path.join(self.output_dir, '%s-blending-model%d' % (self.timestamp, model_cnt))
                    ens_config.append((algo_id, config, model_path, "\n"))
                model_cnt += 1
        ens_info['ensemble_method'] = 'blending'
        ens_info['config'] = ens_config
        ens_info['meta_learner'] = self.meta_method
        return ens_info

    def get_ens_model_details(self):
        model_cnt = 0
        ens_info = {}
        ens_config = []
        output_info = []
        for algo_id in self.stats:
            model_to_eval = self.stats[algo_id]
            for idx, (config, _, _) in enumerate(model_to_eval):
                if not hasattr(self, 'base_model_mask') or self.base_model_mask[model_cnt] == 1:
                    import os
                    model_path = os.path.join(self.output_dir, '%s-blending-model%d' % (self.timestamp, model_cnt))
                    ens_config.append((algo_id, config, model_path, "\n"))
                model_cnt += 1
        ens_info['ensemble_method'] = 'blending'
        ens_info['config'] = ens_config
        ens_info['meta_learner'] = self.meta_method

        for ids, (algo_id, config, model_path) in enumerate(ens_info['config']):
            output_info.append([algo_id, self.meta_method, config, model_path])
        ens_info_mat = pd.DataFrame(output_info, columns=['Algorithm', 'Meta_method', 'Configuration', 'Model_path'])

        import os
        ens_info_save_dir = os.getcwd()
        if not os.path.exists(ens_info_save_dir):
            os.makedirs(ens_info_save_dir)
        import random
        ens_info_mat.to_csv(os.path.join(ens_info_save_dir + "/Ensemble_model_details_" + str(random.randint(1, 10000)) + ".tsv"), sep="\t")
        return 0
