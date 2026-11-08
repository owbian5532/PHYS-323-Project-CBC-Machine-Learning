"""
Variational Quantum Classifier (VQC) with a ZZ feature map  (simulator only)

This is the "trainable circuit" model: unlike Encoding_Comparison.py (which only
computes a quantum *kernel* and lets a classical SVM do the learning), this script
puts trainable parameters directly on the quantum circuit and optimizes them with
a loss function (cross-entropy), exactly the "loss function + training" piece from
Noah's list.

Workflow matches Noah's note: ZZ feature map (encoding) -> variational ansatz
(trainable) -> softmax-style class probabilities -> cross-entropy loss -> optimizer.

Requires: qiskit, qiskit-machine-learning, qiskit-algorithms, scikit-learn, pandas
    pip install --break-system-packages qiskit qiskit-machine-learning qiskit-algorithms

Expects the same train/test files LDA_Split_Script.py already produces:
    cbc_train_lda.csv, cbc_test_lda.csv   (columns: LD1..LD4, diagnosis)

These files are looked up next to this script, then in any subfolder, so it works no
matter which folder your terminal is in. If they don't exist yet, run the LDA split
script first (it needs diagnosed_cbc_data_v4.csv).
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_algorithms.optimizers import COBYLA
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parent      # folder this script lives in


def find_file(name):
    """Look next to this script first, then in any subfolder (e.g. the 'Data transform...' folder)."""
    direct = HERE / name
    if direct.exists():
        return direct
    matches = sorted(HERE.rglob(name), key=lambda p: len(p.parts))
    return matches[0] if matches else direct


TRAIN_FILE = find_file("cbc_train_lda.csv")
TEST_FILE = find_file("cbc_test_lda.csv")
RESULTS_FILE = HERE / "vqc_results.csv"
LABEL_COL = "diagnosis"

REPS_FEATURE_MAP = 2          # matches Encoding_Comparison.py's zz_map
REPS_ANSATZ = 3                # trainable layers; more = more flexible, slower to train
MAX_ITER = 150                 # optimizer iterations
SCALE = np.pi                  # feature range fed into the ZZ map (avoid wrap-around near 2*pi)


def load_split(train_file, test_file, label_col):
    missing = [f.name for f in (train_file, test_file) if not f.exists()]
    if missing:
        sys.exit(
            f"Could not find {', '.join(missing)} in {HERE} or any of its subfolders.\n"
            "Pull the latest from GitHub (the CSVs are in the 'Data transform...' folder), or run "
            "the LDA split script to create them, then rerun this script."
        )
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
    }]).to_csv(RESULTS_FILE, index=False)
    print(f"\nSaved {RESULTS_FILE.name} (total runtime: {train_seconds + predict_seconds:.1f}s)")


if __name__ == "__main__":
    main()
