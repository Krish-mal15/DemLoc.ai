from tqdm import tqdm
from neurodeg_loc_vae import DemLocVAE
from data_processing_loading import dataloader

from torch.optim import Adam
import torch
import torch.nn as nn

from torch_geometric.data import Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DemLocVAE().to(device)
optimizer = Adam(model.parameters(), lr=1e-4)

# BCE Loss + Reconstruction Signal Loss + KL Divergence Loss
def vae_loss(true_label, pred_label, true_signal, pred_signal, mean, var):
    kl_div_loss = -0.5 * torch.sum(1 + var - mean.pow(2) - var.exp(), dim=-1).mean()
    
    recon_crit = nn.MSELoss()
    bce_crit = nn.BCELoss()
    
    recon_loss = recon_crit(pred_signal, true_signal)
    binary_loss = bce_crit(pred_label, true_label)
    
    return binary_loss + recon_loss


num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    
    with tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
        print('')
        for eeg_nodes, eeg_idx, eeg_attr, dem_true_class in pbar:
            eeg_nodes, eeg_idx, eeg_attr, dem_true_class = eeg_nodes.to(device), eeg_idx.to(device), eeg_attr.to(device), dem_true_class.to(device)
            eeg_graph = Data(x=eeg_nodes.squeeze(0), edge_index=eeg_idx.squeeze(0), edge_attr=eeg_attr.squeeze(0))
            
            optimizer.zero_grad()
            
            decoded_dem_pred, recon_signal, mu, log_var, encoded_z = model(eeg_nodes.squeeze(0), eeg_idx.squeeze(0))
            

            loss = vae_loss(dem_true_class.float(), decoded_dem_pred, eeg_graph.x, recon_signal, mu, log_var)
            
            print("Loss: ", loss)
            print("True: ", dem_true_class)
            print("Pred: ", decoded_dem_pred)

            loss.backward()
            optimizer.step()            
            pbar.set_postfix(loss=loss.item())
    
    torch.save(model.state_dict(), f'dem_loc_model_epoch_{epoch+1}_vae.pth')
    