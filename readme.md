# Project 4: Heterogeneous Recommendation Engine v1.0

## Features Implemented
- **Task A (Graph Construction):** Built a tri-partite graph (User-Movie-Actor) using `torch_geometric.data.HeteroData`. Implemented a linear projection layer to align disparate feature dimensions (User: 16, Movie: 32, Actor: 8).
- **Task B (Relational Message Passing):** Developed a `HeteroConv` architecture using GraphSAGE operators that treats `rates` and `stars_in` edges as distinct relational types.
- **Task C (Link Prediction):** Implemented a Link Prediction pipeline with `RandomLinkSplit`, negative sampling, and an evaluation suite measuring Hits@10 and MRR.

## Results
- **Training Epochs:** 50
- **Final Loss:** 0.3246
- **Hits@10:** 0.7580