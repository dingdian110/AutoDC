import numpy as np


def nsav():
    from sklearn.externals import joblib
    import os
    gene_contribution_score = joblib.load(
        os.path.join(os.getcwd() + "/AutoDC_test_results" + "/Feature_Contribution_Score.pkl"))

    return gene_contribution_score


class SelectKBest(object):

    def __init__(
            self,
            score_func='nsav',
            k=200):
        self.score_func = score_func
        self.k = k
        self.values = ''

    def _check_params(self, x):
        if not (self.k == "all" or 0 <= self.k <= x.shape[1]):
            raise ValueError("k should be >=0, <= n_features = %d; got %r. " "Use k='all' to return all features." % (
                x.shape[1], self.k))

    def get_support(self):

        self.values = nsav()
        if self.k == 'all':
            return np.ones(self.values.shape, dtype=bool)
        elif self.k == 0:
            return np.zeros(self.values.shape, dtype=bool)
        else:
            mask = np.zeros(self.values.shape, dtype=bool)
            mask[np.argsort(abs(self.values), kind="mergesort")[-self.k:]] = 1
            return mask

    def fit(self, x):

        if not callable(self.score_func):
            raise TypeError("The score function should be a callable, %s (%s) "
                            "was passed." % (self.score_func, type(self.score_func)))

        self._check_params(x)

        return self

    def transform(self, x):
        mask = self.get_support()
        return x[:, mask]

# if __name__ == "__main__":
#     from autodc.components.feature_engineering.fe_pipeline import FEPipeline
#     X = DataManager().load_train_csv("/home/sky/Desktop/my_test/dev1023/soln-ml-dev/train_data.csv", label_col=0)
#     fe_pipeline = FEPipeline(fe_enabled=False, metric='bal_acc', task_type=0)
#     train_data = fe_pipeline.fit_transform(X)
#     print(train_data.shape)
#     model = SelectKBest(nsav, k=200).transform(np.array(train_data.data[0]))
#     print(model)
