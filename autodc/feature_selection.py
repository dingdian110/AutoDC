import numpy as np
import pandas as pd
from autodc.components.feature_engineering.transformation_graph import DataNode


class FeatureSelectionProcess(object):
    """This class implements the feature selection task. """
    def __init__(self, target_number=100):
        self.target_number = target_number

    def features_from_embedded(self, data: DataNode):
        from lightgbm import LGBMClassifier
        X, y = data.data
        lgb = LGBMClassifier(random_state=1)
        lgb.fit(X, y)
        _importance = lgb.feature_importances_

        mask = np.zeros(X.shape[1], dtype=bool)
        mask[np.argsort(_importance, kind="mergesort")[-self.target_number:]] = 1
        return mask

    def features_from_wrapper(self, data: DataNode):
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression
        X, y = data.data
        model = RFE(estimator=LogisticRegression(solver='lbfgs', max_iter=200, random_state=1),
                    n_features_to_select=self.target_number)
        model.fit(X, y)
        mask = model.get_support()
        return mask

    def features_from_filter(self, data: DataNode):
        from sklearn.feature_selection import SelectKBest, chi2
        X, y = data.data
        model = SelectKBest(chi2, k=self.target_number)
        X_new = model.fit(X, y)
        mask = model.get_support()
        return mask

    def features_from_nsav(self, data: DataNode):
        X, y = data.data
        from scipy import linalg
        U, s, VT = linalg.svd(np.array(X, dtype='float'))
        Gene_Contribution_Score = VT[0, :]

        mask = np.zeros(Gene_Contribution_Score.shape, dtype=bool)
        mask[np.argsort(abs(Gene_Contribution_Score), kind="mergesort")[-self.target_number:]] = 1

        return mask

    def get_features(self, data: DataNode):
        """
        Fit the classifier to given training data.
        :param data: instance of DataNode
        :return: data_new
        """
        embedded = self.features_from_embedded(data)
        filter = self.features_from_filter(data)
        wrapper = self.features_from_wrapper(data)
        nsav = self.features_from_nsav(data)

        mask = np.zeros(embedded.shape, dtype=bool)
        for i in range(len(mask)):
            count = 0
            if embedded[i]:
                count += 1
            if filter[i]:
                count += 1
            if wrapper[i]:
                count += 1
            if nsav[i]:
                count += 1

            if count >= 2:
                mask[i] = True
        return mask


