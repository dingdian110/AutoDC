import os
from autodc.components.feature_engineering.transformations.base_transformer import Transformer
from autodc.components.utils.class_loader import find_components

"""
Load the buildin classifiers.
"""
generator_directory = os.path.split(__file__)[0]
_generator = find_components(__package__, generator_directory, Transformer)
