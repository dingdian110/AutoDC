from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from autodc.components.feature_engineering.transformations.base_transformer import *


class NsavDecomposer(Transformer):
    def __init__(self, target_dim=0.02, random_state=1):
        super().__init__("nsav", 200)
        self.input_type = [NUMERICAL, DISCRETE, CATEGORICAL]
        self.compound_mode = 'only_new'
        self.output_type = NUMERICAL

        self.target_dim = target_dim
        self.random_state = random_state

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        X = input_datanode.data[0]

        from sklearn.externals import joblib
        import os
        gene_contribution_score = joblib.load(
            os.path.join(os.getcwd() + "/AutoDC_test_results/Feature_Contribution_Score.pkl"))

        dim = int(X.shape[1] * self.target_dim)
        if dim < 100:
            dim = 100

        BestKFeatures = np.zeros(gene_contribution_score.shape, dtype=bool)
        BestKFeatures[np.argsort(abs(gene_contribution_score), kind="mergesort")[-dim:]] = 1
        X_new = X[:, BestKFeatures]

        return X_new

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        target_dim = UniformFloatHyperparameter(
            "target_dim", 0, 1, default_value=0.05)
        cs = ConfigurationSpace()
        cs.add_hyperparameter(target_dim)
        return cs
