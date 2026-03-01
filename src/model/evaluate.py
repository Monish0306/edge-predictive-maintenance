import numpy as np
import pandas as pd
import torch
import onnxruntime as ort
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
import json
import sys
import os
sys.path.append('.')

def load_test_data(dataset='FD001'):
    """Load and preprocess test data with RUL labels"""
    import pickle
    from src.data_processing.preprocess import load_dataset, add_labels, create_sequences

    train_path = f'data/raw/train_{dataset}.txt'
    test_path  = f'data/raw/test_{dataset}.txt'
    rul_path   = f'data/raw/RUL_{dataset}.txt'

    train_df, test_df, rul_df = load_dataset(train_path, test_path, rul_path)

    # Get useful sensors from training data
    from sklearn.preprocessing import MinMaxScaler
    columns = train_df.columns.tolist()
    sensor_cols = [c for c in columns if 'sensor' in c]
    useful = [c for c in sensor_cols if train_df[c].std() > 0.001]

    # Fit scaler on train
    scaler = MinMaxScaler()
    scaler.fit(train_df[useful])

    # For test data: RUL file gives TRUE RUL at last cycle
    # Build RUL for test: count backwards from last cycle
    test_df = test_df.copy()
    max_cycles = test_df.groupby('unit_id')['cycle'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    test_df = test_df.merge(max_cycles, on='unit_id')

    # True RUL = RUL from file + remaining cycles
    true_rul_map = {}
    for i, uid in enumerate(sorted(test_df['unit_id'].unique())):
        if i < len(rul_df):
            true_rul_map[uid] = int(rul_df.iloc[i]['RUL'])

    test_df['RUL'] = test_df.apply(
        lambda r: true_rul_map.get(r['unit_id'], 0) +
                  (r['max_cycle'] - r['cycle']), axis=1
    )
    test_df['RUL_clipped'] = test_df['RUL'].clip(upper=125)
    test_df['anomaly'] = (test_df['RUL'] <= 30).astype(int)

    test_df[useful] = scaler.transform(test_df[useful])

    X_test, y_anomaly, y_rul = create_sequences(test_df, useful, seq_len=30)
    return X_test, y_anomaly, y_rul, useful

def evaluate_all_datasets():
    print("="*60)
    print("  FULL TEST SET EVALUATION — ALL 4 DATASETS")
    print("="*60)

    # Load ONNX model
    model_path = 'models/onnx/model_fp32.onnx'
    if not os.path.exists(model_path):
        print("❌ Model not found! Run convert_to_onnx.py first.")
        return

    session = ort.InferenceSession(model_path)
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    all_results = {}

    for ds in datasets:
        print(f"\n[{ds}] Evaluating on test set...")
        try:
            X_test, y_true, y_rul_true, sensors = load_test_data(ds)

            if len(X_test) == 0:
                print(f"  ⚠️  No test sequences for {ds}")
                continue

            # Each dataset has different sensors — reshape to match model
            model_sensors = int(session.get_inputs()[0].shape[2]) if session.get_inputs()[0].shape[2] else 15
            if X_test.shape[2] != model_sensors:
                print(f"  ℹ️  {ds} has {X_test.shape[2]} sensors, model uses {model_sensors}")
                print(f"      Trimming/padding to match model...")
                if X_test.shape[2] > model_sensors:
                    X_test = X_test[:, :, :model_sensors]  # trim extra sensors
                else:
                    # Pad with zeros
                    pad = np.zeros((X_test.shape[0], X_test.shape[1], model_sensors - X_test.shape[2]))
                    X_test = np.concatenate([X_test, pad], axis=2)

            # Run inference
            probs, rul_preds = [], []
            batch_size = 1
            for i in range(0, len(X_test), batch_size):
                batch = X_test[i:i+batch_size].astype(np.float32)
                # Auto-detect input name from model
                input_name = session.get_inputs()[0].name
                out = session.run(None, {input_name: batch})[0]
                probs.extend(out.tolist())

            probs = np.array(probs)
            y_pred = (probs > 0.5).astype(int)

            # Metrics
            f1  = f1_score(y_true, y_pred, zero_division=0)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec  = recall_score(y_true, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_true, probs)
            except:
                auc = 0.0

            acc = (y_pred == y_true).mean()
            cm  = confusion_matrix(y_true, y_pred).tolist()

            result = {
                'dataset': ds,
                'test_samples': int(len(X_test)),
                'accuracy': round(float(acc), 4),
                'f1_score': round(float(f1), 4),
                'precision': round(float(prec), 4),
                'recall': round(float(rec), 4),
                'auc_roc': round(float(auc), 4),
                'confusion_matrix': cm,
                'anomaly_rate_true': round(float(y_true.mean()), 4),
                'anomaly_rate_pred': round(float(y_pred.mean()), 4),
            }

            all_results[ds] = result

            print(f"  Samples   : {len(X_test)}")
            print(f"  Accuracy  : {acc:.4f}")
            print(f"  F1 Score  : {f1:.4f}")
            print(f"  Precision : {prec:.4f}")
            print(f"  Recall    : {rec:.4f}")
            print(f"  AUC-ROC   : {auc:.4f}")
            print(f"  Confusion Matrix:")
            print(f"    TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
            print(f"    FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    # Save results
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  ✅ EVALUATION COMPLETE!")
    print(f"  Results saved: data/processed/evaluation_results.json")
    print(f"{'='*60}")
    return all_results

if __name__ == '__main__':
    evaluate_all_datasets()