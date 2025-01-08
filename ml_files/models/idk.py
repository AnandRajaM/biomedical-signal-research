import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from bilstm import create_model  


def load_data(file_paths):
    data = []
    targets = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)  
        radar_i = df['radar_i'].values
        radar_q = df['radar_q'].values
        tfm_ecg = df['tfm_ecg2'].values
        
        features = np.stack((radar_i, radar_q), axis=1)
        
        data.append(features)
        targets.append(tfm_ecg)

    X = np.vstack(data)
    y = np.concatenate(targets)
    return X, y

file_paths = [r'D:\dev\biomedical-signal-research\datasets\csv_files\datasets_subject_01_to_10_scidata\GDN0001\GDN0001_1_Resting.csv',
              r'D:\dev\biomedical-signal-research\datasets\csv_files\datasets_subject_01_to_10_scidata\GDN0001\GDN0001_2_Valsalva.csv',
              r'D:\dev\biomedical-signal-research\datasets\csv_files\datasets_subject_01_to_10_scidata\GDN0001\GDN0001_3_TiltUp.csv',
              r'D:\dev\biomedical-signal-research\datasets\csv_files\datasets_subject_01_to_10_scidata\GDN0001\GDN0001_4_TiltDown.csv']  
X, y = load_data(file_paths)


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, 2)) 
X_scaled = X_scaled.reshape(-1, 1, 2) 

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_no = 1
results = []

for train_index, val_index in kf.split(X_scaled):
    print(f"Training fold {fold_no}...")

    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y[train_index], y[val_index]

    model = create_model(input_shape=(None, 2))  
    model.fit(X_train, y_train, epochs=10, batch_size=1024, validation_data=(X_val, y_val), verbose=1)

    val_loss = model.evaluate(X_val, y_val, verbose=1)
    print(f"Validation loss for fold {fold_no}: {val_loss}")

    results.append(val_loss)
    fold_no += 1

average_val_loss = np.mean(results)
print(f"Average Validation Loss from K-Fold Cross-Validation: {average_val_loss}")
