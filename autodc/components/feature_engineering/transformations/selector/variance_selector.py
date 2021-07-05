from ConfigSpace.configuration_space import ConfigurationSpace
from autodc.components.feature_engineering.transformations.base_transformer import *


class VarianceSelector(Transformer):
    def __init__(self, threshold=1e-7):
        super().__init__("variance_selector", 9)
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.compound_mode = 'only_new'
        self.threshold = threshold

    def operate(self, input_datanode, target_fields=None):
        from sklearn.feature_selection import VarianceThreshold

        feature_types = input_datanode.feature_types
        X, y = input_datanode.data
        if target_fields is None:
            target_fields = collect_fields(feature_types, self.input_type)
            X_new = X.copy()
        else:
            X_new = X[:, target_fields]

        n_fields = len(feature_types)
        irrevalent_fields = list(range(n_fields))
        for field_id in target_fields:
            irrevalent_fields.remove(field_id)

        is_selected = [True] * len(target_fields)
        if self.model is None:
            self.model = VarianceThreshold(threshold=self.threshold)
            self.model.fit(X_new)

        for idx, var in enumerate(self.model.variances_):
            is_selected[idx] = True if var > self.threshold else False

        irrevalent_types = [feature_types[idx] for idx in irrevalent_fields]
        selected_types = [feature_types[idx] for idx in target_fields if is_selected[idx]]
        selected_types.extend(irrevalent_types)

        _X = self.model.transform(X_new)

        if len(irrevalent_fields) > 0:
            new_X = np.hstack((_X, X[:, irrevalent_fields]))
            if input_datanode.feature_names is not None:
                feature_names = np.hstack(([input_datanode.feature_names[idx] for idx in irrevalent_fields],
                                           [input_datanode.feature_names[idx] for idx in self.model.get_support(True)]))
            else:
                feature_names = None
        else:
            new_X = _X
            if input_datanode.feature_names is not None:
                feature_names = [input_datanode.feature_names[idx] for idx in self.model.get_support(True)]
            else:
                feature_names = None
        new_feature_types = selected_types
        output_datanode = DataNode((new_X, y), new_feature_types, input_datanode.task_type, feature_names=feature_names)
        output_datanode.trans_hist = input_datanode.trans_hist.copy()
        output_datanode.trans_hist.append(self.type)
        output_datanode.enable_balance = input_datanode.enable_balance
        output_datanode.data_balance = input_datanode.data_balance
        self.target_fields = target_fields.copy()

        return output_datanode

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        if optimizer == 'smac':
            cs = ConfigurationSpace()
            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            space = {}
            return space
