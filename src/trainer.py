import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import torch

# =============================================================================
# 4. Training: Trainer Class with Saving/Loading Functionality
# =============================================================================
class Trainer:
    def __init__(self, model, train_loader, criterion, optimizer, device, checkpoint_dir="checkpoints"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self, num_epochs, save_every=5):
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in self.train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
            
            # Save checkpoint every `save_every` epochs
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch + 1, avg_loss)

    def save_checkpoint(self, epoch, loss, filename=None):
        if filename is None:
            filename = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint.get("epoch", 0)
        loss = checkpoint.get("loss", None)
        print(f"Loaded checkpoint from {filename} (Epoch {epoch}, Loss: {loss})")
        return epoch, loss