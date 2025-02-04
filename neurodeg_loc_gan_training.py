import torch
import torch.optim as optim
import torch.nn as nn
from neurodeg_loc_gan_model import ScoringConnectivityGenerator, DementiaConditioningDiscriminator


class DementiaGANTrainer:
    def __init__(
        self,
        generator,
        discriminator,
        device='cuda',
        lr_gen=0.0002,
        lr_disc=0.0002,
        lambda_mmse=0.5  # Weight for MMSE prediction loss
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.lambda_mmse = lambda_mmse
        
        # Initialize optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr_gen, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.999))
        
        # Loss functions
        self.adversarial_loss = nn.BCELoss()
        self.mmse_loss = nn.MSELoss()
        
    def train_discriminator(self, real_graphs, fake_graphs, real_mmse_scores):
        self.optimizer_D.zero_grad()
        batch_size = len(real_graphs)
        
        # Labels for real and fake samples
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        
        # Real samples forward pass
        real_outputs, predicted_mmse = self.discriminator(real_graphs)
        d_loss_real = self.adversarial_loss(real_outputs, real_labels)
        mmse_loss = self.mmse_loss(predicted_mmse, real_mmse_scores)
        
        # Fake samples forward pass
        fake_outputs, _ = self.discriminator(fake_graphs)
        d_loss_fake = self.adversarial_loss(fake_outputs, fake_labels)
        
        # Combined discriminator loss
        d_loss = d_loss_real + d_loss_fake + self.lambda_mmse * mmse_loss
        
        # Backward pass
        d_loss.backward()
        self.optimizer_D.step()
        
        return {
            'disc_loss_total': d_loss.item(),
            'disc_loss_real': d_loss_real.item(),
            'disc_loss_fake': d_loss_fake.item(),
            'mmse_loss': mmse_loss.item()
        }
    
    def train_generator(self, eeg_data, graph_structure):
        self.optimizer_G.zero_grad()
        batch_size = len(eeg_data)
        
        # Generate fake samples
        generated_scores = self.generator(eeg_data)
        
        # Create fake graphs using generated scores
        fake_graphs = self.create_scored_graphs(generated_scores, graph_structure)
        
        # Try to fool discriminator
        fake_outputs, _ = self.discriminator(fake_graphs)
        
        # Generator tries to make discriminator predict real (1)
        g_loss = self.adversarial_loss(fake_outputs, torch.ones(batch_size, 1).to(self.device))
        
        # Backward pass
        g_loss.backward()
        self.optimizer_G.step()
        
        return {'gen_loss': g_loss.item()}
    
    @staticmethod
    def create_scored_graphs(scores, base_graph_structure):
        """
        Creates a graph with the generated importance scores as node features
        while maintaining the original graph structure
        """
        scored_graphs = base_graph_structure.clone()
        scored_graphs.x = scores  # Update node features with generated scores
        return scored_graphs
    
    def train_epoch(self, dataloader, epoch):
        self.generator.train()
        self.discriminator.train()
        
        epoch_metrics = {
            'disc_loss_total': 0.,
            'disc_loss_real': 0.,
            'disc_loss_fake': 0.,
            'mmse_loss': 0.,
            'gen_loss': 0.
        }
        
        for batch_idx, (eeg_data, real_graphs, mmse_scores) in enumerate(dataloader):
            # Move data to device
            eeg_data = eeg_data.to(self.device)
            real_graphs = real_graphs.to(self.device)
            mmse_scores = mmse_scores.to(self.device)
            
            # Generate fake samples
            generated_scores = self.generator(eeg_data)
            fake_graphs = self.create_scored_graphs(generated_scores, real_graphs)
            
            # Train discriminator
            d_metrics = self.train_discriminator(real_graphs, fake_graphs, mmse_scores)
            
            # Train generator
            g_metrics = self.train_generator(eeg_data, real_graphs)
            
            # Update epoch metrics
            for key in epoch_metrics:
                if key in d_metrics:
                    epoch_metrics[key] += d_metrics[key]
                if key in g_metrics:
                    epoch_metrics[key] += g_metrics[key]
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch} [{batch_idx}/{len(dataloader)}]')
                for key, value in epoch_metrics.items():
                    print(f'{key}: {value/(batch_idx+1):.4f}')
        
        # Compute epoch averages
        for key in epoch_metrics:
            epoch_metrics[key] /= len(dataloader)
            
        return epoch_metrics
    
    def get_region_importance(self, eeg_data):
        """
        After training, use this to get region importance scores for new EEG data
        """
        self.generator.eval()
        with torch.no_grad():
            region_scores = self.generator(eeg_data)
        return region_scores

trainer = DementiaGANTrainer(
    generator=ScoringConnectivityGenerator(n_psd_freqs=1025),
    discriminator=DementiaConditioningDiscriminator(),
    device='cuda',
    lambda_mmse=0.5
)

# # Training loop
# num_epochs = 100
# for epoch in range(num_epochs):
#     metrics = trainer.train_epoch(train_dataloader, epoch)
    
#     # Print epoch metrics
#     print(f"\nEpoch {epoch} Summary:")
#     for key, value in metrics.items():
#         print(f"{key}: {value:.4f}")