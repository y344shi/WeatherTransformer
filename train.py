import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Assume X and y are your NumPy arrays from earlier (with shapes as described)
# Convert them to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
# If y has shape (num_samples, 1, feature_size), squeeze if needed for regression tasks.
y_tensor = torch.tensor(y, dtype=torch.float32)

# Create dataset and dataloader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Instantiate model, loss function, and optimizer
model = WeatherTransformer(feature_size=feature_size, forecast_horizon=forecast_horizon)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
