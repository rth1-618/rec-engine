import torch
import torch_geometric.transforms as T
from data_loader import load_graph
from model import GNNModel, LinkPredictor

# 1. Load Data
data = load_graph()

# 2. Task C: Split data and Negative Sampling
# This creates train/val/test sets with "non-existent" edges to practice on
transform = T.RandomLinkSplit(
    num_val=0.1, num_test=0.1,
    neg_sampling_ratio=1.0, 
    edge_types=[('user', 'rates', 'movie')],
    rev_edge_types=[('movie', 'rev_rates', 'user')]
)
train_data, val_data, test_data = transform(data)

# 3. Setup Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNNModel(hidden_channels=64).to(device)
predictor = LinkPredictor().to(device)
optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=0.01)

# 4. Training Loop
def train():
    model.train()
    optimizer.zero_grad()
    
    # --- FIX START ---
    # Move the entire data object to the GPU
    batch = train_data.to(device) 
    # --- FIX END ---
    
    # Forward pass using the data on the correct device
    h_dict = model(batch.x_dict, batch.edge_index_dict)
    
    # Get scores for edges
    edge_label_index = batch['user', 'rates', 'movie'].edge_label_index
    edge_label = batch['user', 'rates', 'movie'].edge_label
    
    # Task C: Prediction & Loss
    preds = predictor(
        h_dict['user'], 
        h_dict['movie'], 
        edge_label_index
    )
    
    loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, edge_label)
    
    loss.backward()
    optimizer.step()
    return loss.item()

lossArr = []
print("Starting Training...")
for epoch in range(1, 51):
    loss = train()
    lossArr.append(loss)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')



import matplotlib.pyplot as plt

plt.plot(range(1, 51), lossArr)
plt.show()



from eval import evaluate

# After the training loop
print("\n--- Final Evaluation ---")
test_data = test_data.to(device)
hits, mrr = evaluate(model, predictor, test_data, k=10)

print(f"Hits@10: {hits:.4f}")
print(f"MRR:     {mrr:.4f}")

print("\n----------DATA-----------")
print(data)

print("\n----------MODEL-----------")
print(model)

print("\n---REAL-WORLD IMPACT ---")
print(f"Success Rate: {hits*100:.1f}% of users found a movie they liked in our Top 10.")
print(f"System Status: Trained on {len(data['user', 'rates', 'movie'].edge_index[0])} interactions.")