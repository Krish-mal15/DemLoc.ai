import torch.nn as nn
import torch
from scipy.signal import welch
from tqdm import tqdm
from neurodeg_anomaly_ae import GraphAnomalyAE
from torch_geometric.data import Data
from autoencoder_anomaly_dataloader import dataloader

device = torch.device('cpu')

def connectivity_loss(pred_adj, true_adj):
    mse = nn.MSELoss()
    loss = mse(pred_adj, true_adj)
    
    return loss

class EEGSignalLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, fs=1000, freq_bands=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.fs = fs 
        self.mse = nn.MSELoss()
        self.freq_bands = freq_bands or {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30)
        }

    def compute_psd(self, signal):
        psd = []
        for channel in signal.detach().cpu().numpy():  
            f, p = welch(channel, fs=self.fs, nperseg=256)
            psd.append((f, p))
        return psd

    def compute_band_power(self, psd):
        band_powers = {}
        for band, (low, high) in self.freq_bands.items():
            band_powers[band] = []
            for f, p in psd:
                idx = (f >= low) & (f < high)
                if idx.any():  # Ensure that there is at least one valid index
                    f_filtered = torch.tensor(f[idx])
                    p_filtered = torch.tensor(p[idx])
                    band_powers[band].append(torch.trapz(p_filtered, f_filtered))
                else:
                    band_powers[band].append(torch.tensor(0.0))  # If no valid indices, append zero
        return band_powers

    def forward(self, true_signal, pred_signal):
        recon_loss = self.mse(pred_signal, true_signal)
        
        # true_psd = self.compute_psd(true_signal)
        # pred_psd = self.compute_psd(pred_signal)
        
        # true_band_power = self.compute_band_power(true_psd)
        # pred_band_power = self.compute_band_power(pred_psd)

        # feature_loss = 0.0
        # for band in self.freq_bands:
        #     true_power = torch.tensor(true_band_power[band])
        #     pred_power = torch.tensor(pred_band_power[band])
        #     feature_loss += self.mse(pred_power, true_power)
        
        # total_loss = self.alpha * recon_loss + self.beta * feature_loss
        return recon_loss


# model = GraphAnomalyAE(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# criterion = EEGSignalLoss()

# num_epochs = 50
# for epoch in range(num_epochs):
#     model.train()
    
#     with tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
#         print('')
#         for eeg_nodes, eeg_idx, eeg_attr in pbar:
#             eeg_nodes, eeg_idx, eeg_attr = eeg_nodes.to(device), eeg_idx.to(device), eeg_attr.to(device)
#             eeg_graph = Data(x=eeg_nodes.squeeze(0), edge_index=eeg_idx.squeeze(0), edge_attr=eeg_attr.squeeze(0))
            
#             optimizer.zero_grad()
            
#             decoded_eeg, latent_features = model(eeg_graph)

#             loss = criterion(eeg_nodes.squeeze(0), decoded_eeg)
            
#             print("Loss: ", loss)

#             loss.backward()
#             optimizer.step()            
#             pbar.set_postfix(loss=loss.item())
    
#     torch.save(model.state_dict(), f'models_anomaly_graph_ae/dem_loc_model_epoch_{epoch+1}_anomaly_ae.pth')
    