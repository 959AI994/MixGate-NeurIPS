from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
from torch_geometric.nn import GATConv
from torch.nn import Linear, LayerNorm

class GATTransformerEncoderLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, concat=False, dropout=0.1, ff_hidden_dim=128):
        super(GATTransformerEncoderLayer, self).__init__()
        
        self.gat = GATConv(in_channels, out_channels, heads=heads, dropout=dropout, concat=concat)
        
        self.ffn = torch.nn.Sequential(
            Linear(out_channels*heads if concat else out_channels, ff_hidden_dim),
            torch.nn.ReLU(),
            Linear(ff_hidden_dim, out_channels*heads if concat else out_channels)
        )
        
        self.norm1 = LayerNorm(out_channels*heads if concat else out_channels)
        self.norm2 = LayerNorm(out_channels*heads if concat else out_channels)
        
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
  
        x_residual = x.clone()
        x = self.gat(x, edge_index)
        x = self.dropout(x)
        x = x + x_residual  
        x = self.norm1(x)   
        
        x_residual = x.clone()
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + x_residual  
        x = self.norm2(x)   
        
        return x

class HierarchicalTransformer(nn.Module):
    def __init__(self, args, modalities=['aig', 'xag', 'xmg', 'mig']):
        super(HierarchicalTransformer, self).__init__()
        self.dim = args.dim_hidden
        self.heads = args.tf_head
        self.depth = args.tf_layer
        self.hier_tf_head = args.hier_tf_head
        self.hier_tf_layer = args.hier_tf_layer
        self.max_hop_once = args.max_hop_once
        self.modalities = modalities
        
        self.hop_tfs = nn.ModuleList([
            GATTransformerEncoderLayer(self.dim*2, self.dim*2, heads=self.hier_tf_head, ff_hidden_dim=self.dim*8)
            for i in range(args.hier_tf_layer) for modal in modalities
        ])
        self.lev_tfs = nn.ModuleList([
            GATTransformerEncoderLayer(self.dim*2, self.dim*2, heads=self.hier_tf_head, ff_hidden_dim=self.dim*8)
            for i in range(args.hier_tf_layer) for modal in modalities
        ])
        self.graph_tfs = nn.ModuleList([
            GATTransformerEncoderLayer(self.dim*2, self.dim*2, heads=self.hier_tf_head, ff_hidden_dim=self.dim*8)
            for i in range(args.hier_tf_layer) for modal in modalities
        ])
        self.mcm_tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.dim * 2, nhead=self.heads, batch_first=True), num_layers=self.depth)
        self.hop_nodes = nn.Parameter(torch.randn(1, args.dim_hidden * 2)) 
        self.subg_nodes = nn.Parameter(torch.randn(1, args.dim_hidden * 2))
        self.graph_nodes = nn.Parameter(torch.randn(1, args.dim_hidden * 2))
        
    def forward(self, g, tokens, masked_tokens, masked_modal='aig'):
        device = next(self.parameters()).device
        mcm_predicted_tokens = torch.zeros(0, self.dim * 2).to(device)
        for batch_id in range(g['batch'].max().item()+1):
            other_modal_tokens = torch.zeros((0, self.dim * 2)).to(device) 
            for modal_k, modal in enumerate(self.modalities):
            
                if modal == masked_modal:
                    batch_masked_tokens = masked_tokens[g['{}_batch'.format(modal)] == batch_id]
                    continue
                
                select_hop = (g['{}_batch'.format(modal)][g['{}_hop_node'.format(modal)]] == batch_id)
                hop_list = g['{}_hop'.format(modal)][select_hop]
                hop_lev_list = g['{}_hop_lev'.format(modal)][select_hop]
                hop_length_list = g['{}_hop_length'.format(modal)][select_hop]
                if hop_lev_list.numel() > 0:
                    max_hop_lev = hop_lev_list.max().item()
                else:
                    max_hop_lev = 0  

                node_tokens = tokens[modal_k]
                all_hop_tokens = torch.zeros((0, self.dim * 2)).to(device)
                all_subg_tokens = torch.zeros((0, self.dim * 2)).to(device)
                for lev in range(max_hop_lev + 1):
                   
                    hop_flag = hop_lev_list == lev
                    no_hops_in_level = hop_list[hop_flag].size(0)
                    if no_hops_in_level == 0:
                        continue
                    level_hop_tokens = torch.zeros((0, self.dim * 2)).to(device)
                 
                    for i in range(0, no_hops_in_level, self.max_hop_once):
                        nodes_in_hop = node_tokens[hop_list[hop_flag][i:i+self.max_hop_once]]
                        nodes_in_hop_flatten = torch.zeros((0, self.dim * 2)).to(device)
                        no_hops_once = min(self.max_hop_once, no_hops_in_level - i)
                        
                        nodes_in_hop_flatten = torch.cat([self.hop_nodes.repeat(no_hops_once, 1), nodes_in_hop_flatten], dim=0)
                        no_nodes_once = 0
                        hop_attn = []
                        for j, length in enumerate(hop_length_list[hop_flag][i:i+self.max_hop_once]):
                            nodes_in_hop_flatten = torch.cat([nodes_in_hop_flatten, nodes_in_hop[j, :length, :]], dim=0)
                            hop_attn.append([j, j])
                            for k in range(length):
                                hop_attn.append([no_hops_once + no_nodes_once + k, j])
                            no_nodes_once += length  
                        hop_attn = torch.tensor(hop_attn, dtype=torch.long).t().contiguous().to(device)   
                        output_nodes_in_hop = self.hop_tfs[modal_k](nodes_in_hop_flatten, hop_attn)
                        hop_tokens = output_nodes_in_hop[:no_hops_once, :]
                        all_hop_tokens = torch.cat([all_hop_tokens, hop_tokens], dim=0)
                        level_hop_tokens = torch.cat([level_hop_tokens, hop_tokens], dim=0)
                    
                    hops_in_subg = torch.cat([self.subg_nodes, level_hop_tokens], dim=0)
                    subg_attn = torch.tensor([[i, 0] for i in range(hops_in_subg.size(0))], dtype=torch.long).t().contiguous().to(device)
                    output_subg_tokens = self.lev_tfs[modal_k](hops_in_subg, subg_attn)
                    subg_tokens = output_subg_tokens[0:1, :]
                    all_subg_tokens = torch.cat([all_subg_tokens, subg_tokens], dim=0)
                    
                
                subg_in_graph = torch.cat([self.graph_nodes, all_subg_tokens], dim=0)
                graph_attn = torch.tensor([[i, 0] for i in range(subg_in_graph.size(0))], dtype=torch.long).t().contiguous().to(device)
                output_graph_tokens = self.graph_tfs[modal_k](subg_in_graph, graph_attn) 
                graph_tokens = output_graph_tokens[0:1, :]
                
                modal_tokens = torch.cat([all_hop_tokens, all_subg_tokens, graph_tokens], dim=0)
                other_modal_tokens = torch.cat([other_modal_tokens, modal_tokens], dim=0)
                
            batch_all_tokens = torch.cat([batch_masked_tokens, other_modal_tokens], dim=0)
            batch_predicted_tokens = self.mcm_tf(batch_all_tokens)
            batch_pred_masked_tokens = batch_predicted_tokens[:batch_masked_tokens.shape[0], :]
            mcm_predicted_tokens = torch.cat([mcm_predicted_tokens, batch_pred_masked_tokens], dim=0)
            
        return mcm_predicted_tokens