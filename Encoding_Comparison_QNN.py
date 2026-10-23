"""
Angle vs ZZ feature map, each feeding a trainable quantum neural network (QNN),
compared with a classical neural network (MLP).  Simulator only.

Pipeline for the quantum models:
    patient -> feature map (angle or ZZ) -> trainable RY + CNOT ansatz
    -> <Z> on each qubit -> trainable linear + softmax layer -> cross-entropy loss
    (each fit is repeated from N_RESTARTS random starts; the lowest loss is kept)

Requires: qiskit, scikit-learn, scipy, numpy, pandas
"""
import numpy as np
import pandas as pd
from functools import reduce
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector, Operator
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score

TRAIN_FILE = "cbc_train_lda.csv"   # or cbc_train_pca.csv
TEST_FILE = "cbc_test_lda.csv"     # or cbc_test_pca.csv
LABEL_COL = "diagnosis"
SEEDS = (0,)                            # CV repeated with these seeds (QNN training is slow; add more if time allows)
ANGLE_SCALES = [0.25, 0.5, 1, 2, np.pi, 4]   # avoid values near 2*pi (angles wrap around)
ZZ_SCALES = [0.25, 0.5, 1, 2, 3, np.pi]
ANSATZ_REPS = 2        # number of (RY layer + CNOT chain) blocks in the trainable part
MAXITER = 200          # optimizer iterations per QNN fit
N_RESTARTS = 2         # random starting points per QNN fit; the lowest-loss one is kept
HIDDEN = (16,)         # classical MLP hidden layer sizes


# ---------------------------------------------------------------- feature maps
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
    """Encode each patient: fill the blanks, simulate, keep the 2^n amplitudes."""
    return np.array([Statevector(fm.assign_parameters(list(row))).data for row in X])


# ------------------------------------------------------------ trainable ansatz
def ry_layer(theta):
    """RY(theta_i) on every qubit as one matrix (qubit 0 = least significant, Qiskit order)."""
    mats = []
    for t in theta:
        c, s = np.cos(t / 2), np.sin(t / 2)
        mats.append(np.array([[c, -s], [s, c]]))
    return reduce(np.kron, mats[::-1])


def entangler(n):
    """Fixed CNOT chain 0->1->2->...  built with Qiskit so qubit ordering matches the encoders."""
    qc = QuantumCircuit(n)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return Operator(qc).data


def ansatz_unitary(theta, E):
    """theta has shape (reps+1, n): RY layer, then (CNOT chain + RY layer) repeated."""
    U = ry_layer(theta[0])
    for r in range(1, theta.shape[0]):
        U = ry_layer(theta[r]) @ E @ U
    return U


def z_signs(n):
    """Matrix (2^n x n): entry [k, j] = +1 if qubit j is 0 in basis state k, else -1."""
    return np.array([[1.0 - 2 * ((k >> j) & 1) for j in range(n)] for k in range(2 ** n)])


def z_expectations(S, U, Zs):
    """<Z_j> on every qubit for every patient after the ansatz. Shape (N, n), values in [-1, 1]."""
    P = np.abs(S @ U.T) ** 2      # measurement probabilities, one row per patient
    return P @ Zs


def softmax(L):
    L = L - L.max(axis=1, keepdims=True)
    e = np.exp(L)
    return e / e.sum(axis=1, keepdims=True)


def unpack(w, shape, n, C):
    """Split the flat parameter vector into ansatz angles, readout weights, readout bias."""
    k = shape[0] * shape[1]
    return w[:k].reshape(shape), w[k:k + n * C].reshape(n, C), w[k + n * C:]


def loss_and_grad(w, S, yi, Zs, E, shape, sw, C):
    """Weighted cross-entropy. Ansatz gradient: parameter-shift. Readout gradient: analytic."""
    eps = 1e-9
    n = shape[1]
    theta, W, b = unpack(w, shape, n, C)
    Z = z_expectations(S, ansatz_unitary(theta, E), Zs)
    Pr = softmax(Z @ W + b)
    N = len(yi)
    idx = np.arange(N)
    loss = -np.sum(sw * np.log(Pr[idx, yi] + eps)) / sw.sum()

    dlog = Pr.copy()                       # d loss / d logits
    dlog[idx, yi] -= 1.0
    dlog *= (sw / sw.sum())[:, None]
    gW = Z.T @ dlog
    gb = dlog.sum(axis=0)
    dZ = dlog @ W.T                        # d loss / d <Z>

    flat = theta.ravel()
    gtheta = np.zeros(flat.size)
    for j in range(flat.size):
        e = np.zeros(flat.size)
        e[j] = np.pi / 2
        Zp = z_expectations(S, ansatz_unitary((flat + e).reshape(shape), E), Zs)
        Zm = z_expectations(S, ansatz_unitary((flat - e).reshape(shape), E), Zs)
        gtheta[j] = np.sum(dZ * (Zp - Zm) / 2)
    return loss, np.concatenate([gtheta, gW.ravel(), gb])


def fit_predict_qnn(Xtr, ytr, Xte, fm, scale, seed):
    """Scale on TRAIN only, encode, train ansatz + readout (best of N_RESTARTS), predict Xte."""
    sc = MinMaxScaler((0, scale)).fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    Str, Ste = states(fm, A), states(fm, B)

    classes = np.unique(ytr)
    C = len(classes)
    yi = np.searchsorted(classes, ytr)
    n = fm.num_qubits

    counts = np.bincount(yi, minlength=C)
    cw = len(yi) / (C * np.maximum(counts, 1))   # 'balanced' class weights
    sw = cw[yi]

    Zs = z_signs(n)
    E = entangler(n)
    shape = (ANSATZ_REPS + 1, n)

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(N_RESTARTS):
        w0 = np.concatenate([rng.uniform(0, 2 * np.pi, size=shape).ravel(),
                             rng.normal(0, 1.0, size=n * C),
                             np.zeros(C)])
        res = minimize(loss_and_grad, w0, args=(Str, yi, Zs, E, shape, sw, C),
                       jac=True, method="L-BFGS-B", options={"maxiter": MAXITER})
        if best is None or res.fun < best.fun:
            best = res

    theta, W, b = unpack(best.x, shape, n, C)
    Z = z_expectations(Ste, ansatz_unitary(theta, E), Zs)
    return classes[np.argmax(Z @ W + b, axis=1)]


def fit_predict_mlp(Xtr, ytr, Xte, seed):
    """Classical neural network baseline (standardized inputs)."""
    ss = StandardScaler().fit(Xtr)
    clf = MLPClassifier(hidden_layer_sizes=HIDDEN, max_iter=2000, random_state=seed)
    return clf.fit(ss.transform(Xtr), ytr).predict(ss.transform(Xte))


def fit_predict(Xtr, ytr, Xte, fm, scale, seed=0):
    """fm=None means classical MLP."""
    if fm is None:
        return fit_predict_mlp(Xtr, ytr, Xte, seed)
    return fit_predict_qnn(Xtr, ytr, Xte, fm, scale, seed)


# ------------------------------------------------------------------ evaluation
def score(y_true, y_pred):
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")


def cv_eval(X, y, fm, scale, k=5):
    """Stratified k-fold CV, repeated over SEEDS. Returns mean (accuracy, macro-F1)."""
    res = []
    for seed in SEEDS:
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
            res.append(score(y[te], fit_predict(X[tr], y[tr], X[te], fm, scale, seed)))
    return tuple(np.mean(res, axis=0))


def main():
    tr = pd.read_csv(TRAIN_FILE)
    te = pd.read_csv(TEST_FILE)
    pcs = [c for c in tr.columns if c != LABEL_COL]
    X, y = tr[pcs].values, tr[LABEL_COL].astype(str).values
    Xte, yte = te[pcs].values, te[LABEL_COL].astype(str).values
    n = X.shape[1]
    print(f"{len(X)} train rows, {len(Xte)} test rows, {n} features -> {n} qubits, "
          f"{len(set(y))} classes")

    models = {"angle-QNN": (angle_map(n), ANGLE_SCALES), "zz-QNN": (zz_map(n), ZZ_SCALES)}

    print("\n--- Cross-validation on TRAIN (used only to pick each scale) ---")
    best = {}
    a, f = cv_eval(X, y, None, 1.0)
    print(f"{'mlp':10} (standardized)  accuracy={a:.3f}  macroF1={f:.3f}")
    for name, (fm, scales) in models.items():
        for s in scales:
            a, f = cv_eval(X, y, fm, s)
            print(f"{name:10} scale={s:5.2f}  accuracy={a:.3f}  macroF1={f:.3f}")
            if name not in best or f > best[name][1]:
                best[name] = (s, f)

    print("\n--- Held-out TEST (scored once, at the CV-chosen scale) ---")
    final = [("mlp", 0.0, *score(yte, fit_predict(X, y, Xte, None, 1.0)))]
    for name, (fm, _) in models.items():
        s = best[name][0]
        final.append((name, s, *score(yte, fit_predict(X, y, Xte, fm, s))))
    for name, s, a, f in final:
        print(f"{name:10} scale={s:5.2f}  accuracy={a:.3f}  macroF1={f:.3f}")
    pd.DataFrame(final, columns=["model", "scale", "accuracy", "macro_f1"]).to_csv(
        "results_qnn.csv", index=False)
    print("\nSaved results_qnn.csv")


if __name__ == "__main__":
    main()
