"""
Angle vs ZZ quantum kernel vs classical RBF  (simulator only)

Requires: qiskit, scikit-learn, numpy, pandas
"""
import numpy as np
import pandas as pd
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

TRAIN_FILE = "cbc_train_lda.csv"   # or cbc_train_pca.csv
TEST_FILE = "cbc_test_lda.csv"     # or cbc_test_pca.csv
LABEL_COL = "diagnosis"
SEEDS = (0, 1, 2)                       # CV repeated with these seeds, then averaged
ANGLE_SCALES = [0.25, 0.5, 1, 2, np.pi, 4]   # avoid values near 2*pi (angles wrap around)
ZZ_SCALES = [0.25, 0.5, 1, 2, 3, np.pi]


def angle_map(n):
    """Recipe 1: one RY rotation per feature; the angle is a blank slot x[i]."""
    x = ParameterVector("x", n)
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(x[i], i)
    return qc


def zz_map(n):
    """Recipe 2: Qiskit's ZZ feature map. reps = repeats; 'full' = every qubit pair linked."""
    return zz_feature_map(n, reps=2, entanglement="full")


def states(fm, X):
    """Fingerprints: fill the blanks with each patient's numbers, simulate, read the 2^n amplitudes."""
    return np.array([Statevector(fm.assign_parameters(list(row))).data for row in X])


def kernel(SA, SB):
    """Similarity table: entry (i, j) = |<fingerprint_i | fingerprint_j>|^2, in [0, 1]."""
    return np.abs(SA.conj() @ SB.T) ** 2


def fit_predict(Xtr, ytr, Xte, fm, scale):
    """Scale on TRAIN only, then fit an SVC and predict Xte. fm=None means classical RBF."""
    if fm is None:      # fair classical baseline: standardize (the quantum kernels need [0, scale])
        ss = StandardScaler().fit(Xtr)
        return SVC(kernel="rbf", class_weight="balanced").fit(
            ss.transform(Xtr), ytr).predict(ss.transform(Xte))
    sc = MinMaxScaler((0, scale)).fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    if fm is None:
        clf = SVC(kernel="rbf", class_weight="balanced").fit(A, ytr)
        return clf.predict(B)
    SA, SB = states(fm, A), states(fm, B)
    clf = SVC(kernel="precomputed", class_weight="balanced").fit(kernel(SA, SA), ytr)
    return clf.predict(kernel(SB, SA))      # rows = test patients, columns = training patients


def score(y_true, y_pred):
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")


def cv_eval(X, y, fm, scale, k=5):
    """Stratified k-fold CV, repeated over SEEDS. Returns mean (accuracy, macro-F1)."""
    res = []
    for seed in SEEDS:
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
            res.append(score(y[te], fit_predict(X[tr], y[tr], X[te], fm, scale)))
    return tuple(np.mean(res, axis=0))


def main():
    tr = pd.read_csv(TRAIN_FILE)
    te = pd.read_csv(TEST_FILE)
    pcs = [c for c in tr.columns if c.startswith("PC")]
    X, y = tr[pcs].values, tr[LABEL_COL].astype(str).values
    Xte, yte = te[pcs].values, te[LABEL_COL].astype(str).values
    n = X.shape[1]
    print(f"{len(X)} train rows, {len(Xte)} test rows, {n} features -> {n} qubits, "
          f"{len(set(y))} classes")

    # RBF baseline: its own scale only changes the input range; gamma adapts automatically
    models = {"angle": (angle_map(n), ANGLE_SCALES), "zz": (zz_map(n), ZZ_SCALES)}

    print("\n--- Cross-validation on TRAIN (used only to pick each scale) ---")
    rows, best = [], {}
    a, f = cv_eval(X, y, None, 1.0)
    print(f"{'rbf':6} (standardized)  accuracy={a:.3f}  macroF1={f:.3f}")
    for name, (fm, scales) in models.items():
        for s in scales:
            a, f = cv_eval(X, y, fm, s)
            print(f"{name:6} scale={s:5.2f}  accuracy={a:.3f}  macroF1={f:.3f}")
            rows.append((name, s, a, f))
            if name not in best or f > best[name][1]:
                best[name] = (s, f)

    print("\n--- Held-out TEST (scored once, at the CV-chosen scale) ---")
    final = [("rbf", 0.0, *score(yte, fit_predict(X, y, Xte, None, 1.0)))]
    for name, (fm, _) in models.items():
        s = best[name][0]
        final.append((name, s, *score(yte, fit_predict(X, y, Xte, fm, s))))
    for name, s, a, f in final:
        print(f"{name:6} scale={s:5.2f}  accuracy={a:.3f}  macroF1={f:.3f}")
    pd.DataFrame(final, columns=["model", "scale", "accuracy", "macro_f1"]).to_csv(
        "results.csv", index=False)
    print("\nSaved results.csv")


if __name__ == "__main__":
    main()
