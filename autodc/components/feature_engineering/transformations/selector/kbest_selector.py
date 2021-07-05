from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, CategoricalHyperparameter, Constant
from autodc.components.feature_engineering.transformations.base_transformer import *
from autodc.utils.logging_utils import get_logger


class KbestSelector(Transformer):
    def __init__(self, kbest=500, score_func='nsav'):
        super().__init__("kbest_selector", 201)
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.compound_mode = 'only_new'
        self.score_func = score_func
        self.kbest = kbest
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)

    def get_score_func(self):
        if self.score_func == 'chi2':
            from sklearn.feature_selection import chi2
            call_func = chi2
        elif self.score_func == 'f_classif':
            from sklearn.feature_selection import f_classif
            call_func = f_classif
        elif self.score_func == 'mutual_info':
            from sklearn.feature_selection import mutual_info_classif
            call_func = mutual_info_classif
        elif self.score_func == 'nsav':
            call_func = 'nsav'
        else:
            raise ValueError("Unknown score function %s!" % str(self.score_func))
        return call_func

    def operate(self, input_datanode, target_fields=None):
        feature_types = input_datanode.feature_types
        X, y = input_datanode.data

        if X.shape[1] < self.kbest:
            return input_datanode
        if target_fields is None:
            target_fields = collect_fields(feature_types, self.input_type)

        X_new = X[:, target_fields]

        n_fields = len(feature_types)
        irrevalent_fields = list(range(n_fields))
        for field_id in target_fields:
            irrevalent_fields.remove(field_id)

        # Because the pipeline guarantees that each feature is positive,
        # clip all values below zero to zero
        if self.score_func == 'chi2' or self.score_func == 'nsav':
            X_new[X_new < 0] = 0.0

        if self.score_func == 'chi2' or self.score_func == 'f_classif' or self.score_func == 'mutual_info':
            if self.model is None:
                from sklearn.feature_selection import SelectKBest
                self.model = SelectKBest(self.get_score_func(), k=self.kbest)
                self.model.fit(X_new, y)
        elif self.score_func == 'nsav':
            if self.model is None:
                from autodc.utils.nsav_selector import SelectKBest
                self.model = SelectKBest(self.get_score_func(), k=self.kbest)
        else:
            sys.exit(1)

        # self.logger.info('kbest_selected_method: %s | Selected top features: %s' % (self.score_func, self.kbest))
        _X = self.model.transform(X_new)
        is_selected = self.model.get_support()

        irrevalent_types = [feature_types[idx] for idx in irrevalent_fields]
        selected_types = [feature_types[idx] for idx in target_fields if is_selected[idx]]
        selected_types.extend(irrevalent_types)

        new_X = np.hstack((_X, X[:, irrevalent_fields]))
        new_feature_types = selected_types
        output_datanode = DataNode((new_X, y), new_feature_types, input_datanode.task_type)
        output_datanode.trans_hist = input_datanode.trans_hist.copy()
        output_datanode.trans_hist.append(self.type)
        output_datanode.enable_balance = input_datanode.enable_balance
        output_datanode.data_balance = input_datanode.data_balance
        self.target_fields = target_fields.copy()

        return output_datanode

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        if optimizer == 'smac':
            kbest = UniformIntegerHyperparameter(
                name="kbest", lower=100, upper=5000, default_value=500)

            score_func = CategoricalHyperparameter(
                name="score_func",
                choices=["chi2", "f_classif", "mutual_info", "nsav"],
                default_value="nsav"
            )
            if dataset_properties is not None:
                # Chi2 can handle sparse data, so we respect this
                if 'sparse' in dataset_properties and dataset_properties['sparse']:
                    score_func = Constant(
                        name="score_func", value="chi2")

            cs = ConfigurationSpace()
            cs.add_hyperparameters([kbest, score_func])

            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            space = {'kbest': hp.uniform('kbest_kbest', 100, 5000),
                     'score_func': hp.choice('kbest_score_func', ['chi2', 'f_classif', 'mutual_info', 'nsav'])}
            return space
