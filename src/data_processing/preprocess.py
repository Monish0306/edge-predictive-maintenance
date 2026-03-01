import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

def load_data(filepath):
    columns = ['unit_id', 'cycle', 'setting1', 'setting2', 'setting3'] + \
              [f'sensor{i}' for i in range(1, 22)]
    df = pd.read_csv(filepath, sep=' ', header=None, names=columns)
    df = df.dropna(axis=1)
    return df

def add_rul(df):
    """Add Remaining Useful Life column"""
    # For each engine, find max cycle = failure point
    max_cycles = df.groupby('unit_id')['cycle'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    df = df.merge(max_cycles, on='unit_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df = df.drop('max_cycle', axis=1)
    return df

def add_anomaly_label(df, threshold=30):
    """If RUL < 30 cycles, label as anomaly (about to fail)"""
    df['anomaly'] = (df['RUL'] <= threshold).astype(int)
    return df

def create_sequences(df, seq_len=30):
    """Create time-series windows for the Transformer"""
    # Select sensor columns only
    sensor_cols = [col for col in df.columns if 'sensor' in col]
    
    X, y = [], []
    
    for unit_id in df['unit_id'].unique():
        unit_data = df[df['unit_id'] == unit_id].sort_values('cycle')
        sensors = unit_data[sensor_cols].values
        labels = unit_data['anomaly'].values
        
        for i in range(len(sensors) - seq_len):
            X.append(sensors[i:i+seq_len])
            y.append(labels[i+seq_len])  # label for next step
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def preprocess_and_save():
    print("Loading data...")
    train_df = load_data('data/raw/train_FD001.txt')
    
    print("Adding RUL...")
    train_df = add_rul(train_df)
    train_df = add_anomaly_label(train_df)
    
    # Select useful sensors (remove constant ones)
    sensor_cols = [col for col in train_df.columns if 'sensor' in col]
    
    # Normalize sensors
    scaler = MinMaxScaler()
    train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])
    
    print("Creating sequences...")
    X, y = create_sequences(train_df, seq_len=30)
    
    print(f"X shape: {X.shape}")  # (samples, 30, num_sensors)
    print(f"y shape: {y.shape}")
    print(f"Anomaly ratio: {y.mean():.2%}")
    
    # Save
    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/X_train.npy', X)
    np.save('data/processed/y_train.npy', y)
    
    # Save scaler for later use
    with open('data/processed/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Data preprocessing complete!")
    return X, y

if __name__ == '__main__':
    preprocess_and_save()