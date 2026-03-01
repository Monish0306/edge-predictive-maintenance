import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import json

def load_dataset(train_path, test_path, rul_path):
    """Load one complete FD00X dataset"""
    columns = ['unit_id', 'cycle', 'setting1', 'setting2', 'setting3'] + \
              [f'sensor{i}' for i in range(1, 22)]

    train = pd.read_csv(train_path, sep=r'\s+', header=None,
                        names=columns, engine='python')
    train = train.dropna(axis=1, how='all')

    test = pd.read_csv(test_path, sep=r'\s+', header=None,
                       names=columns, engine='python')
    test = test.dropna(axis=1, how='all')

    rul = pd.read_csv(rul_path, sep=r'\s+', header=None,
                      names=['RUL'], engine='python')
    return train, test, rul

def add_rul_train(df):
    max_cycles = df.groupby('unit_id')['cycle'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    df = df.merge(max_cycles, on='unit_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df = df.drop('max_cycle', axis=1)
    return df

def add_labels(df, anomaly_threshold=30):
    """Add both anomaly label AND clipped RUL for regression"""
    df['anomaly'] = (df['RUL'] <= anomaly_threshold).astype(int)
    # Clip RUL at 125 — standard practice for this dataset
    df['RUL_clipped'] = df['RUL'].clip(upper=125)
    return df

def remove_constant_sensors(df):
    sensor_cols = [c for c in df.columns if 'sensor' in c]
    useful = [c for c in sensor_cols if df[c].std() > 0.001]
    return useful

def create_sequences(df, sensor_cols, seq_len=30):
    """Create sequences for BOTH anomaly detection AND RUL prediction"""
    X, y_anomaly, y_rul = [], [], []

    for uid in df['unit_id'].unique():
        unit = df[df['unit_id'] == uid].sort_values('cycle')
        sensors = unit[sensor_cols].values
        labels  = unit['anomaly'].values
        ruls    = unit['RUL_clipped'].values

        if len(sensors) < seq_len:
            continue

        for i in range(len(sensors) - seq_len):
            X.append(sensors[i:i+seq_len])
            y_anomaly.append(labels[i+seq_len])
            y_rul.append(ruls[i+seq_len])

    return (np.array(X, dtype=np.float32),
            np.array(y_anomaly, dtype=np.float32),
            np.array(y_rul, dtype=np.float32))

def preprocess_all_datasets():
    print("="*60)
    print("  PREPROCESSING ALL 4 NASA TURBOFAN DATASETS")
    print("="*60)

    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    all_X, all_y_anomaly, all_y_rul = [], [], []
    dataset_info = {}

    os.makedirs('data/processed', exist_ok=True)

    for ds in datasets:
        train_path = f'data/raw/train_{ds}.txt'
        test_path  = f'data/raw/test_{ds}.txt'
        rul_path   = f'data/raw/RUL_{ds}.txt'

        if not os.path.exists(train_path):
            print(f"  ⚠️  {ds} not found, skipping...")
            continue

        print(f"\n[{ds}] Loading...")
        train_df, test_df, rul_df = load_dataset(train_path, test_path, rul_path)

        # Add labels
        train_df = add_rul_train(train_df)
        train_df = add_labels(train_df)

        # Get useful sensors
        useful_sensors = remove_constant_sensors(train_df)

        # Normalize
        scaler = MinMaxScaler()
        train_df[useful_sensors] = scaler.fit_transform(train_df[useful_sensors])

        # Create sequences
        X, y_a, y_r = create_sequences(train_df, useful_sensors)

        print(f"  Engines  : {train_df['unit_id'].nunique()}")
        print(f"  Sensors  : {len(useful_sensors)}")
        print(f"  Sequences: {len(X)}")
        print(f"  Anomaly% : {y_a.mean():.1%}")

        # Save per-dataset
        np.save(f'data/processed/X_{ds}.npy', X)
        np.save(f'data/processed/y_anomaly_{ds}.npy', y_a)
        np.save(f'data/processed/y_rul_{ds}.npy', y_r)

        # Save scaler for this dataset
        with open(f'data/processed/scaler_{ds}.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        dataset_info[ds] = {
            'engines': int(train_df['unit_id'].nunique()),
            'sensors': int(len(useful_sensors)),
            'sequences': int(len(X)),
            'anomaly_rate': round(float(y_a.mean()), 4)
        }

        all_X.append(X)
        all_y_anomaly.append(y_a)
        all_y_rul.append(y_r)

    # Combine all datasets — use common sensors only
    print(f"\n[COMBINED] Merging all datasets...")
    
    # Each dataset has different sensors — save separately, use FD001 as main
    # This is standard practice — FD001 is the benchmark dataset
    X_fd001    = np.load('data/processed/X_FD001.npy')
    y_a_fd001  = np.load('data/processed/y_anomaly_FD001.npy')
    y_r_fd001  = np.load('data/processed/y_rul_FD001.npy')

    # Save main training files (FD001 is primary)
    np.save('data/processed/X_train.npy', X_fd001)
    np.save('data/processed/y_train.npy', y_a_fd001)
    np.save('data/processed/y_rul_train.npy', y_r_fd001)

    num_sensors = X_fd001.shape[2]
    with open('data/processed/num_sensors.txt', 'w') as f:
        f.write(str(num_sensors))

    with open('data/processed/dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  ✅ ALL DATASETS PROCESSED!")
    print(f"  Note: Each dataset has different sensor counts.")
    print(f"  FD001 ({X_fd001.shape[2]} sensors) used as primary training set.")
    print(f"  All 4 datasets available for comparison in dashboard.")
    for ds, info in dataset_info.items():
        print(f"  {ds}: {info['engines']} engines, "
              f"{info['sensors']} sensors, "
              f"{info['sequences']} sequences, "
              f"{info['anomaly_rate']:.1%} anomaly rate")
    print(f"{'='*60}")

    return num_sensors

if __name__ == '__main__':
    preprocess_all_datasets()