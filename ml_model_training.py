"""
3_train_surrogate_model.py
Train neural network surrogate for neutron shielding
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
import json

class ShieldingDataset(Dataset):
    """PyTorch dataset for shielding data"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ShieldingSurrogate(nn.Module):
    """Neural network for flux prediction"""
    
    def __init__(self, input_dim=5, hidden_dims=[128, 64, 32], dropout=0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ModelTrainer:
    """Train and evaluate the surrogate model"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = criterion(predictions, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch.unsqueeze(1))
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, n_epochs=200, lr=0.001):
        """Full training loop"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss = self.validate(val_loader, criterion)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pt'))
    
    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions

def prepare_data(df):
    """Prepare data for training"""
    # Remove rows with missing values
    df = df.dropna(subset=['total_flux'])
    
    # Feature columns
    feature_cols = ['concrete_thick', 'lead_thick', 'poly_thick', 
                    'source_energy', 'source_intensity']
    
    X = df[feature_cols].values
    
    # Use log-transform for flux (better for neural networks)
    y = np.log10(df['total_flux'].values + 1e-10)
    
    return X, y, feature_cols

def plot_training_history(trainer, output_dir='plots'):
    """Plot training curves"""
    Path(output_dir).mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(trainer.train_losses, label='Train Loss')
    plt.plot(trainer.val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions(y_true, y_pred, output_dir='plots'):
    """Plot predicted vs actual"""
    Path(output_dir).mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 10))
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    
    plt.xlabel('True log10(Flux)')
    plt.ylabel('Predicted log10(Flux)')
    plt.title('Surrogate Model Predictions')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate metrics
    mse = np.mean((y_true - y_pred)**2)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
    
    print(f"\nTest Set Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²:  {r2:.6f}")
    
    return {'mse': mse, 'mae': mae, 'r2': r2}

def main():
    # Load data
    print("Loading training data...")
    df = pd.read_csv('training_data.csv')
    print(f"Loaded {len(df)} samples")
    
    # Prepare data
    X, y, feature_cols = prepare_data(df)
    print(f"After cleaning: {len(X)} samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    import pickle
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Create datasets
    train_dataset = ShieldingDataset(X_train_scaled, y_train)
    test_dataset = ShieldingDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = ShieldingSurrogate(
        input_dim=len(feature_cols),
        hidden_dims=[128, 64, 32],
        dropout=0.2
    )
    
    print(f"\nModel architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    print("\nTraining model...")
    trainer = ModelTrainer(model, device=device)
    trainer.train(train_loader, test_loader, n_epochs=200, lr=0.001)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred = trainer.predict(X_test_scaled).flatten()
    
    # Plot results
    plot_training_history(trainer)
    metrics = plot_predictions(y_test, y_pred)
    
    # Save metadata
    metadata = {
        'feature_cols': feature_cols,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'metrics': metrics,
        'model_architecture': {
            'input_dim': len(feature_cols),
            'hidden_dims': [128, 64, 32],
            'dropout': 0.2
        }
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nTraining complete!")
    print("Saved: best_model.pt, scaler.pkl, model_metadata.json")

if __name__ == '__main__':
    main()
