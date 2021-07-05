import os
from autodc.components.utils.constants import FEATURE_TYPES
from autodc.components.utils.utils import find_components, collect_infos
from autodc.components.feature_engineering.transformations.base_transformer import Transformer
from autodc.components.feature_engineering.transformations.generator import _generator
from autodc.components.feature_engineering.transformations.selector import _selector
from autodc.components.feature_engineering.transformations.rescaler import _rescaler
from autodc.components.feature_engineering.transformations.preprocessor import _imb_balancer, _bal_balancer
from autodc.components.feature_engineering.transformations.empty_transformer import EmptyTransformer
from autodc.components.feature_engineering.transformations.continous_discretizer import KBinsDiscretizer
from autodc.components.feature_engineering.transformations.discrete_categorizer import DiscreteCategorizer

"""
Load the build-in transformers.
"""
transformers_directory = os.path.split(__file__)[0]
_transformers = find_components(__package__, transformers_directory, Transformer)

for sub_pkg in ['generator', 'preprocessor', 'rescaler', 'selector']:
    tmp_directory = os.path.split(__file__)[0] + '/%s' % sub_pkg
    transformers = find_components(__package__ + '.%s' % sub_pkg, tmp_directory, Transformer)
    for key, val in transformers.items():
        if key not in _transformers:
            _transformers[key] = val
        else:
            raise ValueError('Repeated Transformer ID: %s!' % key)

_type_infos, _params_infos = collect_infos(_transformers, FEATURE_TYPES)

_preprocessor1 = {'continous_discretizer': KBinsDiscretizer}
_preprocessor2 = {'discrete_categorizer': DiscreteCategorizer}
_preprocessor = {}
for key in _generator:
    # if key not in ['arithmetic_transformer', 'binary_transformer', 'lda_decomposer', 'pca_decomposer', 'kitchen_sinks']:
    if key not in ['arithmetic_transformer', 'binary_transformer', 'lda_decomposer']:
        _preprocessor[key] = _generator[key]
for key in _selector:
    # if key not in ['rfe_selector', 'variance_selector', 'percentile_selector', 'percentile_selector_regression']:
    if key not in ['rfe_selector', 'variance_selector']:
        _preprocessor[key] = _selector[key]

_preprocessor1['empty'] = EmptyTransformer
_preprocessor2['empty'] = EmptyTransformer
_preprocessor['empty'] = EmptyTransformer
_generator['empty'] = EmptyTransformer
_bal_balancer['empty'] = EmptyTransformer
_imb_balancer['empty'] = EmptyTransformer
_selector['empty'] = EmptyTransformer
_rescaler['empty'] = EmptyTransformer
