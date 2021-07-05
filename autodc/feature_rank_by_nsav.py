from autodc.components.feature_engineering.transformations.base_transformer import *


class FeatureRank(object):
    """This class implements the feature selection task. """

    def nsav_caculation(self, data: DataNode):
        X, y = data.data
        from scipy import linalg
        U, s, VT = linalg.svd(np.array(X, dtype='float'))
        Gene_Contribution_Score = VT[0, :]

        from sklearn.externals import joblib
        import os
        nsav_save_dir = os.getcwd() + "/AutoDC_test_results"
        if not os.path.exists(nsav_save_dir):
            os.makedirs(nsav_save_dir)
        joblib.dump(Gene_Contribution_Score, os.path.join(nsav_save_dir + "/Feature_Contribution_Score.pkl"))

        gene_names = data.feature_names
        ens_info = list()
        for idx, gene_id in enumerate(gene_names):
            ens_info.append([gene_id, abs(Gene_Contribution_Score[idx])])

        ens_info_mat = pd.DataFrame(sorted(ens_info, key=lambda x:x[1], reverse=True), columns=['Gene_name', 'Importance_scores'])
        ens_info_save_dir = os.getcwd()
        if not os.path.exists(ens_info_save_dir):
            os.makedirs(ens_info_save_dir)
        ens_info_mat.to_csv(os.path.join(
            ens_info_save_dir + "/Feature_importance_score_by_NSAV" + ".tsv"), sep="\t")

        return 0

# if __name__ == "__main__":
#     from autodc.utils.data_manager import DataManager
#     from autodc.components.feature_engineering.bio_fe_pipeline import FEPipeline
#
#     input_node = DataManager().load_train_csv(
#         "/home/sky/Desktop/my_test/dev1023/soln-ml-dev/train_data_head100_fs500.csv", label_col=0)
#     fe_pipeline = FEPipeline(fe_enabled=False, metric='bal_acc', task_type=0)
#     input_node = fe_pipeline.fit_transform(input_node)
#
#     fe = FeatureRank()
#     a = fe.nsav_caculation(input_node)
