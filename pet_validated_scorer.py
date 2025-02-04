from torch_geometric.nn import GIN
import torch.nn as nn

# PET scans should be correlated just by pearson correlation, because scan color indicates amyloid levels
# Anatomically connected using brain atlas

# Loss will have the EEG scorer use PET derived amyloid levels.
# The below class develops an embedding for the PET graph utilizing 3D color (conv) embeddings
# so that regional scores will more closely reflect the behavior of the PET scans it is validated by
class PETValidatorContext(nn.Module):
    def __init__(self, n_pet_dim, latent_dim):
        super().__init__(self)
        
        self.conv_embed = nn.Conv3d(in_channels=n_pet_dim, out_channels=latent_dim, kernel_size=3)
        
        self.pet_embedding = GIN(
            in_channels=latent_dim,
            hidden_channels=latent_dim * 2,
            out_channels=latent_dim,
            num_layers=4
        )
        
    def forward(self, pet_scan, pet_scan_idx, pet_scan_attn):
        pet_latent_embeds = self.conv_embed(pet_scan)
        # Graph embeddings will ensure brain activity levels working in the same functional connectivity manner as they should.
        pet_graph_embeds = self.pet_embedding(pet_latent_embeds, pet_scan_idx, pet_scan_attn)
        return pet_graph_embeds