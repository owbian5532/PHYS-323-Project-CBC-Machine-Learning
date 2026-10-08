import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

INPUT_FILE = "diagnosed_cbc_data_v4.csv"
OUTPUT_FILE = "cbc_reduced.csv"
TARGET_ROWS = 50
N_PCS = 4
PC_COLS = [f"PC{i+1}" for i in range(N_PCS)]

# --- Load: skip the header row, then address columns by position ---
df = pd.read_csv(INPUT_FILE, header=0)

X_raw = df.iloc[:, :14]                    # columns 1-14: numeric
labels = df.iloc[:, 14].astype(str).str.strip()  # column 15: diagnosis

# Drop any rows with missing values so PCA doesn't fail
mask = X_raw.notna().all(axis=1) & labels.notna()
X_raw = X_raw[mask].astype(float).reset_index(drop=True)
labels = labels[mask].reset_index(drop=True)
print(f"Rows used: {len(X_raw)} (dropped {(~mask).sum()} with missing values)")

# --- PCA: 14 numeric columns -> 2 components ---
X_scaled = StandardScaler().fit_transform(X_raw)
pca = PCA(n_components=N_PCS)
Z = pca.fit_transform(X_scaled)
print("Explained variance per PC:", pca.explained_variance_ratio_.round(3))
print("Total explained variance:", round(pca.explained_variance_ratio_.sum(), 3))

data = pd.DataFrame(Z, columns=PC_COLS)
data["diagnosis"] = labels

# --- Allocate exactly TARGET_ROWS clusters across diagnoses, proportional to size ---
counts = data["diagnosis"].value_counts()
quota = TARGET_ROWS * counts / counts.sum()
k = np.maximum(1, np.floor(quota).astype(int))   # at least 1 per diagnosis
k = np.minimum(k, counts)                         # never more clusters than rows

# Adjust so the total is exactly TARGET_ROWS (when possible)
remainder = (quota - np.floor(quota)).sort_values(ascending=False)
while k.sum() < TARGET_ROWS:
    candidates = [d for d in remainder.index if k[d] < counts[d]]
    if not candidates:
        break
    k[candidates[0]] += 1
    remainder = remainder.drop(candidates[0])
    if remainder.empty:
        remainder = (quota - np.floor(quota)).sort_values(ascending=False)
while k.sum() > TARGET_ROWS:
    d = k[k > 1].idxmax()
    k[d] -= 1

print("\nClusters per diagnosis:")
print(k.to_string())

# --- K-means within each diagnosis ---
parts = []
for dx, group in data.groupby("diagnosis"):
    n_clusters = int(k[dx])
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    km.fit(group[PC_COLS])
    centers = pd.DataFrame(km.cluster_centers_, columns=PC_COLS)
    centers["diagnosis"] = dx
    parts.append(centers)

reduced = pd.concat(parts, ignore_index=True)
reduced = reduced[PC_COLS + ["diagnosis"]]

reduced.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved {len(reduced)} rows x {reduced.shape[1]} columns to {OUTPUT_FILE}")
print(reduced.head(10))
