from tqdm import tqdm
from neurodeg_context_scoring_v2 import EEGScorer, DementiaPredLossContext
from data_processing_loading import dataloader

from torch.optim import Adam
import torch
import torch.nn as nn

from torch_geometric.data import Data
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r".*torch.*")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

main_scorer = EEGScorer().to(device)
context_dem_pred = DementiaPredLossContext().to(device)

optimizer_scorer = Adam(main_scorer.parameters(), lr=1e-4)
optimizer_pred_context = Adam(context_dem_pred.parameters(), lr=1e-4)

criterion = nn.BCELoss()

num_epochs = 50

for epoch in range(num_epochs):
    main_scorer.train()
    context_dem_pred.train()
  
    with tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
        print('')
        for node, idx, attr, mmse, label in pbar:
            node, idx, attr, mmse, label = node.to(device), idx.to(device), attr.to(device), mmse.to(device), label.to(device)
            eeg_graph = Data(x=node.squeeze(0), edge_index=idx.squeeze(0), edge_attr=attr.squeeze(0))
            
            optimizer_scorer.zero_grad()
            optimizer_pred_context.zero_grad()
            
            regional_scores = main_scorer(eeg_graph, mmse)
            context_dem = context_dem_pred(regional_scores)
            
            context_dem = torch.tensor(context_dem, dtype=torch.float, requires_grad=True)
            label = torch.tensor(label, dtype=torch.float, requires_grad=True)

            loss = criterion(context_dem.squeeze(0), label)
            loss.backward()
            
            print(loss)
            print(context_dem)
            print(regional_scores)
            
            optimizer_scorer.step()
            optimizer_pred_context.step()
            
            pbar.set_postfix(loss=loss.item())
    
    torch.save(main_scorer.state_dict(), f'dem_loc_model_main_scorer_epoch_{epoch+1}_v2.pth')
    torch.save(context_dem_pred.state_dict(), f'dem_loc_model_context_epoch_{epoch+1}_v2pth')
    