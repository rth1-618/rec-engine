import pandas as pd
import torch
from torch_geometric.data import HeteroData

def load_graph():
    data = HeteroData()
    
    # 1. Load Data (Using a sample for speed)
    ratings = pd.read_csv('data/rating.csv').sample(100000)
    tags = pd.read_csv('data/tag.csv').sample(50000)  # Represents Actors/Metadata
    
    # 2. Create Mappings
    u_map = {id: i for i, id in enumerate(ratings['userId'].unique())}
    m_map = {id: i for i, id in enumerate(pd.concat([ratings['movieId'], tags['movieId']]).unique())}
    a_map = {tag: i for i, tag in enumerate(tags['tag'].unique())} # Actor/Tag map

    # 3. Task A: Features (Alignment preparation)
    data['user'].x = torch.randn(len(u_map), 16) 
    data['movie'].x = torch.randn(len(m_map), 32)
    data['actor'].x = torch.randn(len(a_map), 8) # Different dimension!

    # 4. Task B: Edges (Relational Message Passing)
    # User -> Movie
    u_idx = [u_map[x] for x in ratings['userId']]
    m_idx_r = [m_map[x] for x in ratings['movieId']]
    data['user', 'rates', 'movie'].edge_index = torch.tensor([u_idx, m_idx_r], dtype=torch.long)
    data['movie', 'rev_rates', 'user'].edge_index = torch.tensor([m_idx_r, u_idx], dtype=torch.long)

    # Actor -> Movie
    a_idx = [a_map[x] for x in tags['tag']]
    m_idx_t = [m_map[x] for x in tags['movieId']]
    data['actor', 'stars_in', 'movie'].edge_index = torch.tensor([a_idx, m_idx_t], dtype=torch.long)
    data['movie', 'rev_stars_in', 'actor'].edge_index = torch.tensor([m_idx_t, a_idx], dtype=torch.long)
    
    return data
