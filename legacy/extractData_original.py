import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder

# Path to dataset
DATASET_PATH = "UCI HAR Dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "train", "Inertial Signals")
TEST_PATH = os.path.join(DATASET_PATH, "test", "Inertial Signals")

# 9 raw signal files
SIGNALS = [
    "body_acc_x_", "body_acc_y_", "body_acc_z_",
    "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
    "total_acc_x_", "total_acc_y_", "total_acc_z_"
]

# -------- Load signals --------
def load_signals(path, split):
    data = []
    for sig in SIGNALS:
        filename = os.path.join(path, sig + split + ".txt")
        arr = np.loadtxt(filename)
        data.append(arr)
    # (signals, samples, timesteps) -> (samples, timesteps, signals)
    data = np.transpose(np.array(data), (1, 2, 0))
    return data

X_train = load_signals(TRAIN_PATH, "train")
X_test = load_signals(TEST_PATH, "test")

print("Train shape:", X_train.shape)  # (7352, 128, 9)
print("Test shape:", X_test.shape)    # (2947, 128, 9)

# -------- Load labels --------
y_train = np.loadtxt(os.path.join(DATASET_PATH, "train", "y_train.txt"))
y_test = np.loadtxt(os.path.join(DATASET_PATH, "test", "y_test.txt"))

# One-hot encode labels
enc = OneHotEncoder(sparse_output=False)
y_train = enc.fit_transform(y_train.reshape(-1, 1))
y_test = enc.transform(y_test.reshape(-1, 1))

print("y_train shape:", y_train.shape)  # (7352, 6)
print("y_test shape:", y_test.shape)    # (2947, 6)

# -------- Normalize --------
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
