from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch_geometric.data import Data
from .data_utils import construct_node_feature
from .dag_utils import return_order_info
        
class OrderedData(Data):
    def __init__(self, edge_index=None, x=None, y=None, tt_pair_index = None, tt_dis = None, \
                 forward_level=None, forward_index=None, backward_level=None, backward_index=None):
        super().__init__()
        self.edge_index = edge_index
        self.tt_pair_index = tt_pair_index
        self.x = x
        self.prob = y
        self.tt_dis = tt_dis
        self.forward_level = forward_level
        self.forward_index = forward_index
        self.backward_level = backward_level
        self.backward_index = backward_index

    def __inc__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key:
            if 'aig' in key:
                return len(self.aig_x)  
            elif 'xag' in key:
                return len(self.xag_x)  
            elif 'xmg' in key:
                return len(self.xmg_x)  
            elif 'mig' in key:
                return len(self.mig_x)  
            else:
                return self.num_nodes  

        if 'batch' in key:
            return 1

        return 0  
    
    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'forward_index' in key or 'backward_index' in key:
            return 0
        elif 'edge_index' in key:
            return 1
        elif 'tt_pair_index' in key:
            return 1
        elif key == 'tt_pair_index' or key == 'connect_pair_index':
            return 1
        else:
            return 0

def parse_pyg_mlpgate(x, edge_index, tt_dis, tt_pair_index, \
                        prob, \

                        ):
    
    x_torch = torch.LongTensor(x)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    tt_dis = torch.tensor(tt_dis)
    tt_pair_index = torch.tensor(tt_pair_index, dtype=torch.long).contiguous()

    forward_level, forward_index, backward_level, backward_index = return_order_info(edge_index, x_torch.size(0))

    forward_level = torch.tensor(forward_level)
    backward_level = torch.tensor(backward_level)

    forward_index = torch.tensor(forward_index)
    backward_index = torch.tensor(backward_index)

    graph = OrderedData(x=x_torch, edge_index=edge_index, y=prob, tt_pair_index = tt_pair_index, tt_dis = tt_dis,
                        forward_level=forward_level, forward_index=forward_index, 
                        backward_level=backward_level, backward_index=backward_index)
    graph.use_edge_attr = False
    
    graph.prob = torch.tensor(prob).reshape((len(x)))

    return graph