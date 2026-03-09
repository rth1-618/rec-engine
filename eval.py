import torch

@torch.no_grad()
def evaluate(model, predictor, data, k=10):
    model.eval()
    predictor.eval()
    
    # 1. Get Node Embeddings
    h_dict = model(data.x_dict, data.edge_index_dict)
    
    # 2. Get the positive and negative edge indices
    # edge_label == 1 are real connections, 0 are fake ones created by RandomLinkSplit
    edge_label_index = data['user', 'rates', 'movie'].edge_label_index
    edge_label = data['user', 'rates', 'movie'].edge_label
    
    # Get scores from the predictor
    logits = predictor(h_dict['user'], h_dict['movie'], edge_label_index)
    
    # Separate positive and negative scores
    pos_scores = logits[edge_label == 1]
    neg_scores = logits[edge_label == 0]
    
    # Hits@K: For each positive edge, how many negative edges does it beat?
    # Simplified version for Hackathon V1.0: 
    # Check what % of positive edges scored higher than the average negative edge
    hits = (pos_scores > neg_scores.mean()).float().mean().item()
    
    # MRR: Mean Reciprocal Rank
    # We rank the positive scores against the negatives
    combined = torch.cat([pos_scores, neg_scores])
    ranks = torch.argsort(combined, descending=True)
    
    # Find where the positive ones ended up
    mrr = 0
    for i in range(len(pos_scores)):
        # Find index of the i-th positive score in the sorted list
        rank = (ranks == i).nonzero(as_tuple=True)[0].item() + 1
        mrr += 1.0 / rank
    mrr /= len(pos_scores)
    
    return hits, mrr
