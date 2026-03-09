import torch
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

class GNNModel(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict()
        
        # Task B: HeteroConv handles multiple relations
        self.conv = HeteroConv({
            ('user', 'rates', 'movie'): SAGEConv((-1, -1), hidden_channels),
            ('movie', 'rev_rates', 'user'): SAGEConv((-1, -1), hidden_channels),
            ('actor', 'stars_in', 'movie'): SAGEConv((-1, -1), hidden_channels),
            ('movie', 'rev_stars_in', 'actor'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='sum')
        
    def forward(self, x_dict, edge_index_dict):
    # Task A: Feature Alignment
        for node_type, x in x_dict.items():
            if node_type not in self.lin_dict:
                # Create the layer AND move it to the GPU immediately
                self.lin_dict[node_type] = Linear(-1, 64).to(x.device) 
        
            x_dict[node_type] = self.lin_dict[node_type](x).relu()
        
    # Task B: Relational Message Passing
        return self.conv(x_dict, edge_index_dict)


class LinkPredictor(torch.nn.Module):
    def forward(self, x_user, x_movie, edge_label_index):
        # edge_label_index[0] contains User IDs
        # edge_label_index[1] contains Movie IDs
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)

