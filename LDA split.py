"""
Makes cbc_train_<METHOD>.csv and cbc_test_<METHOD>.csv from the raw Kaggle CBC file.
METHOD = "lda" (supervised, keeps class information; recommended) or "pca" (unsupervised).

"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

INPUT_FILE = "diagnosed_cbc_data_v4.csv"

N_PCS = 4
METHOD = "lda"          # "lda" or "pca"
TEST_SIZE = 0.2
SEED = 0

df = pd.read_csv(INPUT_FILE, header=0)
X = df.iloc[:, :14]
y = df.iloc[:, 14].astype(str).str.strip()

ok = X.notna().all(axis=1) & y.notna()
X, y = X[ok].astype(float).reset_index(drop=True), y[ok].reset_index(drop=True)
print(f"Rows used: {len(X)} (dropped {(~ok).sum()} with missing values)")

counts = y.value_counts()
print("\nPatients per diagnosis:\n", counts.to_string())
rare = counts[counts < 2].index
if len(rare):
    print(f"\nWARNING: dropping diagnoses with only 1 patient (cannot split): {list(rare)}")
    keep = ~y.isin(rare)
    X, y = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

sc = StandardScaler().fit(Xtr)                     # learned from TRAIN only
Atr = sc.transform(Xtr)
if METHOD == "lda":
    reducer = LinearDiscriminantAnalysis(n_components=N_PCS).fit(Atr, ytr)   # uses labels, TRAIN only
    print("\nClass-separation per component:", reducer.explained_variance_ratio_[:N_PCS].round(3))
else:
    reducer = PCA(n_components=N_PCS).fit(Atr)                                # TRAIN only
    print("\nExplained variance per PC:", reducer.explained_variance_ratio_.round(3))

cols = [f"LD{i+1}" for i in range(N_PCS)]
for name, Xs, ys in [(f"cbc_train_{METHOD}.csv", Xtr, ytr), (f"cbc_test_{METHOD}.csv", Xte, yte)]:
    out = pd.DataFrame(reducer.transform(sc.transform(Xs)), columns=cols)
    out["diagnosis"] = ys.values
    out.to_csv(name, index=False)
    print(f"Saved {name}: {len(out)} rows")
