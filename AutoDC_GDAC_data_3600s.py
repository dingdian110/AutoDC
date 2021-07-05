# -*- coding: UTF-8 -*-
import argparse
import os
import sys
from time import *
from sklearn.metrics import accuracy_score
from autodc.utils.data_manager import DataManager
from autodc.components.feature_engineering.fe_pipeline import FEPipeline
from autodc.bio_estimator import Classifier
from autodc.adaptive_dimension_redection import AdaptiveDimensionReduction

sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument('--time_limit', type=int, default=3600)
parser.add_argument('--eval_type', type=str, default='holdout', choices=['holdout', 'cv', 'partial'])
parser.add_argument('--ens_method', default='ensemble_selection',
                    choices=['none', 'bagging', 'blending', 'stacking', 'ensemble_selection'])
parser.add_argument('--mode', type=str, default='combined')
parser.add_argument('--n_jobs', type=int, default=1)

args = parser.parse_args()

time_limit = args.time_limit
eval_type = args.eval_type
n_jobs = args.n_jobs
ensemble_method = args.ens_method
if ensemble_method == 'none':
    ensemble_method = None

save_dir = './AutoDC_test_results'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print('==============================================================')
print('==> Start...')
print('==> Evaluate with Budget %d' % args.time_limit)
begin_time = time()
print('==> Load the train data...')
dm = DataManager()
train_data = dm.load_train_csv('./data/LUAD_and_LUSC/LUAD_LUSC_train.csv', label_col=0)
# train_data = dm.load_train_csv('./train_data_head100_fs500.csv', label_col=0)
print('==> Preprocess the train data...')
fe_pipeline = FEPipeline(fe_enabled=False, metric='bal_acc', task_type=0)
train_data = fe_pipeline.fit_transform(train_data)
print('==> Adaptive dimension reduction...')
adr = AdaptiveDimensionReduction()
select_genes = adr.rank_features(train_data)
train_data = adr.dimension_reduction(train_data, select_genes)
print('==> Search the best models...')
fs_time = time() - begin_time
time_limit = args.time_limit - fs_time
clf = Classifier(time_limit=time_limit,
                 output_dir=save_dir,
                 ensemble_method='ensemble_selection',
                 per_run_time_limit=400,
                 include_preprocessors=['nsav_decomposer'],
                 evaluation=eval_type,
                 include_algorithms=['random_forest', 'passive_aggressive', 'liblinear_svc', 'gradient_boosting'],
                 metric='bal_acc',
                 n_jobs=n_jobs)
clf.fit(train_data, opt_strategy='combined')
print('==> Output the ensemble model...')
if ensemble_method == 'ensemble_selection':
    clf.get_ens_model_details()
print(clf.get_ens_model_info())
print('==> Load the test data...')
test_data = dm.load_test_csv('./data/LUAD_and_LUSC/LUAD_LUSC_test.csv', has_label=True, label_col=0)
# test_data = dm.load_test_csv('./test_data_head100_fs500.csv', has_label=True, label_col=0)
test_data = fe_pipeline.transform(test_data)
test_data = adr.dimension_reduction(test_data, select_genes)
print('==> Predict...')
pred = clf.predict(test_data)
print("Accracy on Test data: ", accuracy_score(test_data.data[1], pred))
end_time = time()
run_time = end_time - begin_time
print('Total spend time：', run_time)
print('==>Finish!')
print('==============================================================')
