import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import mlflow
import mlflow.pytorch
import os, sys, json
sys.path.append('.')

from src.model.transformer_model import PredMaintenanceTransformer, count_parameters

def train_model():
    CONFIG = {
        'seq_len': 30,
        'batch_size': 64,
        'epochs': 25,
        'learning_rate': 0.001,
        'd_model': 32,
        'nhead': 4,
        'num_layers': 2,
        'dropout': 0.1,
        'rul_loss_weight': 0.3,  # balance between anomaly + RUL loss
    }

    print("="*55)
    print("  DUAL-HEAD TRANSFORMER TRAINING")
    print("  Task 1: Anomaly Detection")
    print("  Task 2: RUL Prediction")
    print("="*55)

    # Load data
    X = np.load('data/processed/X_train.npy')
    y_anomaly = np.load('data/processed/y_train.npy')

    # Load RUL if available
    rul_path = 'data/processed/y_rul_train.npy'
    if os.path.exists(rul_path):
        y_rul = np.load(rul_path)
        y_rul = y_rul / 125.0  # normalize 0-1
    else:
        y_rul = np.zeros_like(y_anomaly)

    num_sensors = X.shape[2]
    print(f"\nData: {X.shape} | Sensors: {num_sensors}")
    print(f"Anomaly rate: {y_anomaly.mean():.1%}")

    # Tensors
    X_t = torch.FloatTensor(X)
    ya_t = torch.FloatTensor(y_anomaly)
    yr_t = torch.FloatTensor(y_rul)

    dataset = TensorDataset(X_t, ya_t, yr_t)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG['batch_size'])

    # Model
    model = PredMaintenanceTransformer(
        num_sensors=num_sensors,
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    )
    print(f"Parameters: {count_parameters(model):,}")

    # Loss functions
    pos_weight = torch.tensor([(y_anomaly==0).sum() / max((y_anomaly==1).sum(), 1)])
    anomaly_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    rul_criterion     = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4)

    # MLflow
    mlflow.set_experiment("predictive_maintenance")

    with mlflow.start_run(run_name="dual_head_transformer"):
        mlflow.log_params(CONFIG)
        mlflow.log_param("num_sensors", num_sensors)
        mlflow.log_param("model_type", "DualHead_Transformer")
        mlflow.log_param("tasks", "anomaly_detection + RUL_prediction")

        best_val_acc = 0

        for epoch in range(CONFIG['epochs']):
            # ── TRAIN ──
            model.train()
            train_loss, correct, total = 0, 0, 0

            for Xb, ya_b, yr_b in train_loader:
                optimizer.zero_grad()
                anomaly_logit, rul_pred = model(Xb)

                loss_a = anomaly_criterion(anomaly_logit, ya_b)
                loss_r = rul_criterion(rul_pred, yr_b)
                loss   = loss_a + CONFIG['rul_loss_weight'] * loss_r

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                pred = (torch.sigmoid(anomaly_logit) > 0.5).float()
                correct += (pred == ya_b).sum().item()
                total   += len(ya_b)

            train_acc = correct / total
            avg_loss  = train_loss / len(train_loader)

            # ── VALIDATE ──
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            rul_errors = []

            with torch.no_grad():
                for Xb, ya_b, yr_b in val_loader:
                    anomaly_logit, rul_pred = model(Xb)
                    loss_a = anomaly_criterion(anomaly_logit, ya_b)
                    loss_r = rul_criterion(rul_pred, yr_b)
                    val_loss += (loss_a + CONFIG['rul_loss_weight'] * loss_r).item()

                    pred = (torch.sigmoid(anomaly_logit) > 0.5).float()
                    val_correct += (pred == ya_b).sum().item()
                    val_total   += len(ya_b)

                    rul_err = torch.abs(rul_pred - yr_b).mean().item() * 125
                    rul_errors.append(rul_err)

            val_acc  = val_correct / val_total
            avg_vl   = val_loss / len(val_loader)
            mean_rul_err = np.mean(rul_errors)

            scheduler.step(avg_vl)

            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("train_acc",  train_acc, step=epoch)
            mlflow.log_metric("val_loss",   avg_vl,    step=epoch)
            mlflow.log_metric("val_acc",    val_acc,   step=epoch)
            mlflow.log_metric("rul_mae_cycles", mean_rul_err, step=epoch)

            print(f"Epoch {epoch+1:2d}/{CONFIG['epochs']} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Acc: {train_acc:.3f} | "
                  f"Val Acc: {val_acc:.3f} | "
                  f"RUL MAE: {mean_rul_err:.1f} cycles")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs('models/saved', exist_ok=True)
                torch.save(model.state_dict(), 'models/saved/best_model.pth')
                print(f"  ✅ Best model saved!")

        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.log_param("num_sensors_final", num_sensors)
        mlflow.pytorch.log_model(model, "dual_head_model")

        print(f"\n{'='*55}")
        print(f"  TRAINING COMPLETE!")
        print(f"  Best Val Accuracy : {best_val_acc:.4f}")
        print(f"{'='*55}")

if __name__ == '__main__':
    train_model()