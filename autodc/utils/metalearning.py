import os
import functools

from autosklearn.smbo import AutoMLSMBO
from autosklearn.data.xy_data_manager import XYDataManager
from autosklearn.util.backend import create
from autosklearn.util import pipeline, StopWatch

from autodc.datasets.utils import load_data
from autodc.components.feature_engineering.transformations.empty_transformer import *

from autodc.components.feature_engineering.transformations.selector.generic_univariate_selector import *
from autodc.components.feature_engineering.transformations.selector.liblinear_based_selector import *
from autodc.components.feature_engineering.transformations.selector.percentile_selector import *
from autodc.components.feature_engineering.transformations.selector.extra_trees_based_selector import *

from autodc.components.feature_engineering.transformations.rescaler.quantile_transformer import *

from autodc.components.feature_engineering.transformations.generator.kernel_pca import *
from autodc.components.feature_engineering.transformations.generator.kitchen_sinks import *
from autodc.components.feature_engineering.transformations.generator.polynomial_generator import *
from autodc.components.feature_engineering.transformations.generator.pca_decomposer import *
from autodc.components.feature_engineering.transformations.generator.fast_ica_decomposer import *
from autodc.components.feature_engineering.transformations.generator.nystronem_sampler import *
from autodc.components.feature_engineering.transformations.generator.svd_decomposer import *
from autodc.components.feature_engineering.transformations.generator.kitchen_sinks import *
from autodc.components.feature_engineering.transformations.generator.random_trees_embedding import *
from autodc.components.feature_engineering.transformations.generator.svd_decomposer import *
from autodc.components.feature_engineering.transformations.generator.feature_agglomeration_decomposer import *


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    return NewCls


def get_meta_learning_configs(X, y, task_type, dataset_name='default', metric='accuracy', num_cfgs=5):
    if X is None or y is None:
        X, y, _ = load_data(dataset_name)
    backend = create(temporary_directory=None,
                     output_directory=None,
                     delete_tmp_folder_after_terminate=False,
                     delete_output_folder_after_terminate=False,
                     shared_mode=True)
    dm = XYDataManager(X, y, None, None, task_type, None, dataset_name)

    configuration_space = pipeline.get_configuration_space(dm.info,
                                                           include_estimators=None,
                                                           exclude_estimators=None,
                                                           include_preprocessors=None,
                                                           exclude_preprocessors=None)

    watcher = StopWatch()
    name = os.path.basename(dm.name)
    watcher.start_task(name)

    def reset_data_manager(max_mem=None):
        pass

    automlsmbo = AutoMLSMBO(config_space=configuration_space,
                            dataset_name=dataset_name,
                            backend=backend,
                            total_walltime_limit=1e5,
                            func_eval_time_limit=1e5,
                            memory_limit=1e5,
                            metric=metric,
                            watcher=watcher,
                            metadata_directory='components/meta_learning/meta_resource',
                            num_metalearning_cfgs=num_cfgs)
    automlsmbo.reset_data_manager = reset_data_manager
    automlsmbo.task = task_type
    automlsmbo.datamanager = dm
    configs = automlsmbo.get_metalearning_suggestions()
    return configs


def get_trans_from_str(str):
    tran = Empty
    if str == 'quantile_transformer':
        tran = QuantileTransformation
    elif str == 'select_rates':
        tran = GenericUnivariateSelector
    elif str == 'select_percentile_classification':
        tran = PercentileSelector
    elif str == 'liblinear_svc_preprocessor':
        tran = LibLinearBasedSelector
    elif str == 'extra_trees_preproc_for_classification':
        tran = ExtraTreeBasedSelector
    elif str == 'kernel_pca':
        tran = ExtraTreeBasedSelector
    elif str == 'extra_trees_preproc_for_classification':
        tran = ExtraTreeBasedSelector
    elif str == 'kernel_pca':
        tran = KernelPCA
    elif str == 'kitchen_sinks':
        tran = KitchenSinks
    elif str == 'nystroem_sampler':
        tran = NystronemSampler
    elif str == 'random_trees_embedding':
        tran = RandomTreesEmbeddingTransformation
    elif str == 'select_percentile_classification':
        tran = PercentileSelector
    elif str == 'truncatedSVD':
        tran = SvdDecomposer
    elif str == 'feature_agglomeration':
        tran = FeatureAgglomerationDecomposer
    elif str == 'fast_ica':
        tran = FastIcaDecomposer
    elif str == 'pca':
        tran = PcaDecomposer
    elif str == 'polynomial':
        tran = PolynomialTransformation

    return tran
