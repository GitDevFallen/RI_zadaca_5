import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Učitavanje i priprema podataka
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Podjela na trening i test skupove
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizacija podataka
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PyTorch tensori
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Definicija modela
class HousingFFNN(nn.Module):
    def __init__(self):
        super(HousingFFNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)
    
# Inicijalizacija modela, gubitka i optimizatora
model = HousingFFNN()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trening modela
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass i optimizacija
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluacija modela
model.eval()
with torch.no_grad():
    y_predictions = model(X_test_tensor).squeeze().numpy()
    y_test_np = y_test_tensor.squeeze().numpy()
    
    mse = mean_squared_error(y_test_np, y_predictions)
    mae = mean_absolute_error(y_test_np, y_predictions)
    r2 = r2_score(y_test_np, y_predictions)
    
    print("\nRezultati na testnom skupu:")
    print(f'MSE: {mse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R^2 Score: {r2:.4f}')