import abc
import typing
from autodc.components.feature_engineering.transformations import _transformers, _type_infos, _params_infos
from autodc.components.feature_engineering.transformation_graph import DataNode, TransformationGraph
from autodc.utils.functions import is_unbalanced_dataset
from autodc.utils.logging_utils import get_logger


class Optimizer(object, metaclass=abc.ABCMeta):
    def __init__(self, name, task_type, datanode, seed=1):
        self.name = name
        self._seed = seed
        self.root_node = datanode.copy_()
        self.incumbent = self.root_node
        self.task_type = task_type
        self.graph = TransformationGraph()
        self.graph.add_node(self.root_node)
        self.time_budget = None
        self.maximum_evaluation_num = None
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)

        self.if_bal = False if is_unbalanced_dataset(data_node=datanode) else True

    @abc.abstractmethod
    def optimize(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def iterate(self):
        pass

    def get_incumbent(self):
        return self.incumbent

    def get_incumbent_path(self):
        ref_node = self.get_incumbent()
        path_ids = self.graph.get_path_nodes(ref_node)
        self.logger.info('The path ids: %s' % str(path_ids))
        edge_attrs = list()
        for node_id in path_ids[1:]:
            edge = self.graph.get_edge(self.graph.input_edge_dict[node_id])
            self.logger.info('Transformation: %s - %d' % (edge.transformer.name, edge.transformer.type))
            edge_attrs.append(edge.transformer.get_attributes())
        return edge_attrs

    def apply(self, data_node: DataNode, ref_node: DataNode, phase='test'):
        path_ids = self.graph.get_path_nodes(ref_node)
        self.logger.info('The path ids: %s' % str(path_ids))
        if len(path_ids) == 0:
            path_ids = [0]
        inputnode = self.graph.get_node(path_ids[0])
        inputnode.set_values(data_node)
        edge_attrs = list()

        for node_id in path_ids[1:]:
            input_node_list = list()
            for input_id in self.graph.input_data_dict[node_id]:
                inputnode = self.graph.get_node(input_id)
                input_node_list.append(inputnode)
            inputnode = input_node_list[0] if len(input_node_list) == 1 else input_node_list

            edge = self.graph.get_edge(self.graph.input_edge_dict[node_id])
            self.logger.info('Transformation: %s - %d' % (edge.transformer.name, edge.transformer.type))
            edge_attr = str(edge.transformer.get_attributes())
            self.logger.info('Edge attribute: %s' % edge_attr)
            edge_attrs.append(edge_attr)
            outputnode = edge.transformer.operate(inputnode, edge.target_fields)
            self.logger.info('%s => %s' % (str(inputnode.shape), str(outputnode.shape)))
            self.graph.get_node(node_id).set_values(outputnode)
        output_node = self.graph.get_node(path_ids[-1]).copy_()
        self.logger.info('returned shape: %s' % str(output_node.shape))
        self.logger.info('Attribute path: %s' % ','.join(edge_attrs))
        return output_node

    def get_pipeline(self, ref_node: DataNode):
        path_ids = self.graph.get_path_nodes(ref_node)
        edge_attrs = list()

        for node_id in path_ids[1:]:
            edge = self.graph.get_edge(self.graph.input_edge_dict[node_id])
            edge_attr = str(edge.transformer.get_attributes())
            edge_attrs.append(edge_attr)
        return edge_attrs

    def get_available_transformations(self, node: DataNode, trans_types: typing.List):
        return self.get_transformations(list(set(node.feature_types)), trans_types)

    @staticmethod
    def get_transformations(feat_type: str or list[str], trans_types: typing.List):
        if isinstance(feat_type, str):
            feat_type = [feat_type]

        trans_ids = list()
        for _type in feat_type:
            trans_ids.extend(_type_infos[_type])
        trans_ids = list(set(trans_ids))
        transformers = list()

        for id in trans_ids:
            if _transformers[id]().type not in trans_types:
                continue

            params = _params_infos[id]
            if len(params) == 0:
                transformers.append(_transformers[id]())
            else:
                for param in params:
                    transformer = _transformers[id](param=param)
                    transformers.append(transformer)
        return transformers
