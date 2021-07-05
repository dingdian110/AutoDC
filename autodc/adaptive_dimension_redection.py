import warnings
import abc
import numpy as np
import os
warnings.filterwarnings("ignore")

from autodc.components.feature_engineering.transformations.base_transformer import *
from autodc.components.feature_engineering.transformation_graph import DataNode
from sklearn.externals import joblib


class AdaptiveDimensionReduction(object, metaclass=abc.ABCMeta):
    """This class implements the adaptive dimension reduction task. """

    def __init__(self):
        self.bool_select_gene_list = list()

    def rank_features(self, data: DataNode):
        X, y = data.data
        from scipy import linalg
        U, s, VT = linalg.svd(np.array(X, dtype='float'))
        Gene_Contribution_Score = VT[0, :]

        gene_names = data.feature_names
        ens_info = list()
        for idx, gene_id in enumerate(gene_names):
            ens_info.append([gene_id, abs(Gene_Contribution_Score[idx])])

        ens_info_mat = pd.DataFrame(sorted(ens_info, key=lambda x: x[1], reverse=True),
                                    columns=['Gene_name', 'Importance_scores'])
        ens_info_save_dir = os.getcwd()
        if not os.path.exists(ens_info_save_dir):
            os.makedirs(ens_info_save_dir)
        ens_info_mat.to_csv(os.path.join(
            ens_info_save_dir + "/Feature_importance_score_by_NSAV" + ".tsv"), sep="\t")

        half_size = int(len(ens_info) / 2) - 1
        threshold = ens_info_mat.iloc[half_size, 1]
        genelist = ens_info_mat[(ens_info_mat['Importance_scores'] >= threshold)].iloc[:, 0]

        self.bool_select_gene_list = [x in list(genelist) for x in gene_names]

        nsav_save_dir = os.getcwd() + "/AutoDC_test_results"
        if not os.path.exists(nsav_save_dir):
            os.makedirs(nsav_save_dir)
        joblib.dump(Gene_Contribution_Score[self.bool_select_gene_list], os.path.join(nsav_save_dir + "/Feature_Contribution_Score.pkl"))

        return self.bool_select_gene_list

    def dimension_reduction(self, node, select_genes):
        raw_dataframe = node.data[0]
        new_dataframe = raw_dataframe[:, select_genes]
        feature_types = [i for indx, i in enumerate(node.feature_types) if select_genes[indx] == True]
        feature_names = [i for indx, i in enumerate(node.feature_names) if select_genes[indx] == True]
        new_node = DataNode(data=[new_dataframe, node.data[1]], feature_type=feature_types, feature_names=feature_names)

        return new_node


