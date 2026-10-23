"""
Variational Quantum Classifier (VQC) with a ZZ feature map  (simulator only)
"""
import time
import numpy as np
import pandas as pd

from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_algorithms.optimizers import COBYLA
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

TRAIN_FILE = "cbc_train_lda.csv"
TEST_FILE = "cbc_test_lda.csv"
LABEL_COL = "diagnosis"

REPS_FEATURE_MAP = 2          
REPS_ANSATZ = 3                # more = more flexible, slower to train
MAX_ITER = 150                 # optimizer iterations
SCALE = np.pi                 


def load_split(train_file, test_file, label_col):
    tr = pd.read_csv(train_file)
    te = pd.read_csv(test_file)
    feat_cols = [c for c in tr.columns if c != label_col]
    Xtr, Xte = tr[feat_cols].values, te[feat_cols].values
    ytr, yte = tr[label_col].astype(str).values, te[label_col].astype(str).values
    return Xtr, ytr, Xte, yte, feat_cols


def build_vqc(n_qubits, n_classes):
    """ZZ feature map (encoding) + RealAmplitudes ansatz (trainable) -> VQC."""
    feature_map = zz_feature_map(n_qubits, reps=REPS_FEATURE_MAP, entanglement="full")
    ansatz = real_amplitudes(n_qubits, reps=REPS_ANSATZ, entanglement="full")

    log = {"iter": 0}

    def callback(weights, loss):
        log["iter"] += 1
        if log["iter"] % 10 == 0 or log["iter"] == 1:
            print(f"  iter {log['iter']:4d}  loss={loss:.4f}")

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        loss="cross_entropy",          # softmax + cross-entropy under the hood
        optimizer=COBYLA(maxiter=MAX_ITER),
        callback=callback,
    )
    return vqc


def main():
    Xtr_raw, ytr_raw, Xte_raw, yte_raw, feat_cols = load_split(TRAIN_FILE, TEST_FILE, LABEL_COL)
    n_qubits = Xtr_raw.shape[1]

    # Scale features on TRAIN only, into [0, SCALE] (same convention as Encoding_Comparison.py)
    scaler = MinMaxScaler((0, SCALE)).fit(Xtr_raw)
    Xtr, Xte = scaler.transform(Xtr_raw), scaler.transform(Xte_raw)

    # VQC needs numeric/one-hot labels, not strings
    encoder = LabelEncoder().fit(ytr_raw)
    ytr, yte = encoder.transform(ytr_raw), encoder.transform(yte_raw)
    n_classes = len(encoder.classes_)

    print(f"{len(Xtr)} train rows, {len(Xte)} test rows, {n_qubits} features -> {n_qubits} qubits, "
          f"{n_classes} classes: {list(encoder.classes_)}")

    vqc = build_vqc(n_qubits, n_classes)

    print("\n--- Training VQC (ZZ feature map + trainable ansatz) ---")
    train_start = time.perf_counter()
    vqc.fit(Xtr, ytr)
    train_seconds = time.perf_counter() - train_start
    print(f"Training time: {train_seconds:.1f}s")

    print("\n--- Evaluating on held-out TEST ---")
    predict_start = time.perf_counter()
    ypred = vqc.predict(Xte)
    predict_seconds = time.perf_counter() - predict_start
    acc = accuracy_score(yte, ypred)
    f1 = f1_score(yte, ypred, average="macro")
    print(f"vqc_zz  accuracy={acc:.3f}  macroF1={f1:.3f}  "
          f"predict_time={predict_seconds:.1f}s")

    pd.DataFrame([{
        "model": "vqc_zz",
        "accuracy": acc,
        "macro_f1": f1,
        "train_seconds": round(train_seconds, 2),
        "predict_seconds": round(predict_seconds, 2),
        "total_seconds": round(train_seconds + predict_seconds, 2),
    }]).to_csv("vqc_results.csv", index=False)
    print(f"\nSaved vqc_results.csv (total runtime: {train_seconds + predict_seconds:.1f}s)")


if __name__ == "__main__":
    main()
