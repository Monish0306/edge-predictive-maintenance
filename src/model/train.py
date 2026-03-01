import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import mlflow
import mlflow.pytorch
import os
import sys
sys.path.append('.')

from src.model.transformer_model import PredMaintenanceTransformer

def train_model():
    # ── CONFIG ──────────────────────────────────────────
    CONFIG = {
        'seq_len': 30,
        'batch_size': 64,
        'epochs': 20,
        'learning_rate': 0.001,
        'd_model': 32,
        'nhead': 4,
        'num_layers': 2,
        'dropout': 0.1,
    }
    
    # ── LOAD DATA ────────────────────────────────────────
    print("Loading preprocessed data...")
    X = np.load('data/processed/X_train.npy')
    y = np.load('data/processed/y_train.npy')
    
    num_sensors = X.shape[2]
    print(f"Data: {X.shape}, Sensors: {num_sensors}")
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # 80/20 split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'])
    
    # ── MODEL ────────────────────────────────────────────
    model = PredMaintenanceTransformer(
        num_sensors=num_sensors,
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    )
    
    # Handle class imbalance (fewer anomalies than normal)
    pos_weight = torch.tensor([(y == 0).sum() / (y == 1).sum()])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    # ── MLFLOW TRACKING ──────────────────────────────────
    mlflow.set_experiment("predictive_maintenance")
    
    with mlflow.start_run():
        # Log config
        mlflow.log_params(CONFIG)
        mlflow.log_param("num_sensors", num_sensors)
        
        best_val_acc = 0
        
        for epoch in range(CONFIG['epochs']):
            # ── TRAIN ──
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == y_batch).sum().item()
                total += len(y_batch)
            
            train_acc = correct / total
            avg_train_loss = train_loss / len(train_loader)
            
            # ── VALIDATE ──
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    val_correct += (predicted == y_batch).sum().item()
                    val_total += len(y_batch)
            
            val_acc = val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            
            # Log to MLflow
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            
            print(f"Epoch {epoch+1:2d}/{CONFIG['epochs']} | "
                  f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.3f} | "
                  f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.3f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs('models/saved', exist_ok=True)
                torch.save(model.state_dict(), 'models/saved/best_model.pth')
                print(f"  ✓ Best model saved! Val Acc: {val_acc:.3f}")
        
        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.pytorch.log_model(model, "model")
        
        print(f"\nTraining complete! Best Val Accuracy: {best_val_acc:.3f}")
        return model, num_sensors

if __name__ == '__main__':
    train_model()