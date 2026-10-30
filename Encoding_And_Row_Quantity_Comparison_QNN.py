"""
Angle vs ZZ feature map, each feeding a trainable quantum neural network (QNN),
compared with a classical neural network (MLP).  Simulator only.

THIS VERSION runs the way the project actually plans to run it:
  * each quantum model trains on N_TRAIN_ROWS completely random training rows
  * the whole experiment is repeated over many seeds (a different random 50 rows each time)
  * the scale is picked by cross-validation on those same 50 rows
  * every model is scored on the FULL held-out test set, and results are averaged over seeds
  * the MLP is run on the same 50 rows (fair comparison) and once on all rows (reference)

Pipeline for the quantum models:
    patient -> feature map (angle or ZZ) -> trainable RY + CNOT ansatz
    -> <Z> on each qubit -> trainable linear + softmax layer -> cross-entropy loss
    (each fit is repeated from N_RESTARTS random starts; the lowest loss is kept)

Requires: qiskit, scikit-learn, scipy, numpy, pandas
"""
import os
# Tiny matrices run fastest on ONE thread per process; many threads just fight each other.
# (Must be set before numpy is imported. The seeds are spread across processes instead.)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import reduce
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector, Operator
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

# With only 50 rows, a rare diagnosis may have fewer than 5 members, which makes
# scikit-learn print a harmless warning about the CV folds. Hide it.
warnings.filterwarnings("ignore", message="The least populated class")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

TRAIN_FILE = "cbc_train_lda.csv"
TEST_FILE = "cbc_test_lda.csv"
LABEL_COL = "diagnosis"

ROW_COUNTS = [25, 50, 75, 100, 150, 200]   # training-set sizes to sweep
SEEDS = tuple(range(10))          # each seed = a different random ordering of the training rows
ANGLE_SCALES = [2, np.pi, 5]      # the 3 most-picked angle scales from the last run
ZZ_SCALES = [0.25, 0.35, 0.75]    # the 3 most-picked ZZ scales from the last run
ANSATZ_REPS = 2        # number of (RY layer + CNOT chain) blocks in the trainable part
MAXITER = 200          # optimizer iterations per QNN fit
N_RESTARTS = 3         # random starting points per QNN fit (cheap with only 50 rows)
HIDDEN = (16,)         # classical MLP hidden layer sizes
# Seeds run in parallel, one per CPU core. Leave one core free so the computer stays usable.
N_WORKERS = max(1, min(len(SEEDS), (os.cpu_count() or 2) - 1))


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


def ry_layer_derivs(theta):
    """Derivative of ry_layer(theta) with respect to each angle, as a list (one matrix per qubit)."""
    mats, dmats = [], []
    for t in theta:
        c, s = np.cos(t / 2), np.sin(t / 2)
        mats.append(np.array([[c, -s], [s, c]]))
        dmats.append(0.5 * np.array([[-s, -c], [c, -s]]))
    out = []
    for q in range(len(theta)):
        m = list(mats)
        m[q] = dmats[q]
        out.append(reduce(np.kron, m[::-1]))
    return out


def loss_and_grad(w, S, yi, Zs, E, shape, sw, C):
    """Weighted cross-entropy with an EXACT gradient.
    Readout gradient: closed-form formula.
    Ansatz gradient: differentiate the circuit matrix one angle at a time. Because the circuit is
    a chain of layers, we reuse the layers before and after the one being differentiated, so each
    angle costs one small matrix product instead of two full circuit re-runs (parameter-shift)."""
    eps = 1e-9
    R, n = shape
    theta, W, b = unpack(w, shape, n, C)

    # layers in the order they are applied: L0, then (E, L1), (E, L2), ...
    L = [ry_layer(theta[k]) for k in range(R)]
    M = [L[0]] + [L[k] @ E for k in range(1, R)]
    prefix = [np.eye(2 ** n)]                       # prefix[k] = M[k-1] ... M[0]
    for k in range(R - 1):
        prefix.append(M[k] @ prefix[k])
    suffix = [None] * R                             # suffix[k] = M[R-1] ... M[k+1]
    suffix[R - 1] = np.eye(2 ** n)
    for k in range(R - 2, -1, -1):
        suffix[k] = suffix[k + 1] @ M[k + 1]
    U = M[R - 1] @ prefix[R - 1]

    A = S @ U.T                                     # amplitudes after the circuit
    P = np.abs(A) ** 2
    Z = P @ Zs
    Pr = softmax(Z @ W + b)
    N = len(yi)
    idx = np.arange(N)
    loss = -np.sum(sw * np.log(Pr[idx, yi] + eps)) / sw.sum()

    dlog = Pr.copy()                                # d loss / d logits
    dlog[idx, yi] -= 1.0
    dlog *= (sw / sw.sum())[:, None]
    gW = Z.T @ dlog
    gb = dlog.sum(axis=0)
    G = (dlog @ W.T) @ Zs.T                         # d loss / d (probability of each outcome)

    gtheta = np.zeros((R, n))
    Ac = A.conj()
    for k in range(R):
        for q, dL in enumerate(ry_layer_derivs(theta[k])):
            dM = dL if k == 0 else dL @ E
            dU = suffix[k] @ dM @ prefix[k]
            dP = 2 * np.real(Ac * (S @ dU.T))
            gtheta[k, q] = np.sum(G * dP)
    return loss, np.concatenate([gtheta.ravel(), gW.ravel(), gb])


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


def cv_eval(X, y, fm, scale, seed, k=5):
    """One round of stratified k-fold CV on the sampled rows. Returns mean (accuracy, macro-F1)."""
    res = []
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        res.append(score(y[te], fit_predict(X[tr], y[tr], X[te], fm, scale, seed)))
    return tuple(np.mean(res, axis=0))


def run_seed(seed, Xall, yall, Xte, yte):
    """ONE seed: for each row count, train the MLP and every (encoding, scale) combination on the
    same rows, and score each on the full test set. No cross-validation: scales are fixed."""
    n = Xall.shape[1]
    models = {"angle-QNN": (angle_map(n), ANGLE_SCALES), "zz-QNN": (zz_map(n), ZZ_SCALES)}
    rows, log = [], []
    # One random ordering per seed; the first k rows are the sample, so smaller samples are
    # contained in larger ones (this makes the size comparison less noisy).
    order = np.random.default_rng(seed).permutation(len(Xall))

    def record(model, scale, n_rows, n_cls, y_pred):
        acc, f1 = score(yte, y_pred)
        rows.append({"model": model, "scale": round(float(scale), 3), "n_rows": n_rows, "seed": seed,
                     "classes_in_sample": n_cls, "accuracy": acc, "macro_f1": f1})

    for n_rows in ROW_COUNTS:
        X, y = Xall[order[:n_rows]], yall[order[:n_rows]]
        n_cls = len(set(y))
        record("mlp", 0.0, n_rows, n_cls, fit_predict(X, y, Xte, None, 1.0, seed))
        for name, (fm, scales) in models.items():
            for s in scales:
                record(name, s, n_rows, n_cls, fit_predict(X, y, Xte, fm, s, seed))
        log.append(f"  {n_rows} rows done ({n_cls} of {len(set(yall))} diagnoses present)")
    return seed, rows, f"seed {seed} finished\n" + "\n".join(log)


def main():
    tr = pd.read_csv(TRAIN_FILE)
    te = pd.read_csv(TEST_FILE)
    pcs = [c for c in tr.columns if c != LABEL_COL]
    Xall, yall = tr[pcs].values, tr[LABEL_COL].astype(str).values
    Xte, yte = te[pcs].values, te[LABEL_COL].astype(str).values
    print(f"{len(Xall)} train rows available, sweeping {ROW_COUNTS} rows; {len(Xte)} test rows, "
          f"{Xall.shape[1]} qubits, {len(SEEDS)} seeds on {N_WORKERS} worker process(es)", flush=True)

    full_acc, full_f1 = score(yte, fit_predict(Xall, yall, Xte, None, 1.0, 0))
    print(f"Reference, MLP on all {len(Xall)} rows: accuracy={full_acc:.3f}  macroF1={full_f1:.3f}\n",
          flush=True)

    runs, t0 = [], time.perf_counter()
    if N_WORKERS > 1:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = [pool.submit(run_seed, s, Xall, yall, Xte, yte) for s in SEEDS]
            for fut in as_completed(futures):
                seed, rows, log = fut.result()
                runs.extend(rows)
                print(f"[{time.perf_counter() - t0:5.0f}s] {log}\n", flush=True)
    else:
        for s in SEEDS:
            seed, rows, log = run_seed(s, Xall, yall, Xte, yte)
            runs.extend(rows)
            print(f"[{time.perf_counter() - t0:5.0f}s] {log}\n", flush=True)

    runs_df = pd.DataFrame(runs)
    runs_df.to_csv("results_row_sweep_runs.csv", index=False)
    summary = (runs_df.groupby(["model", "scale", "n_rows"])[["accuracy", "macro_f1"]]
               .agg(["mean", "std"]).round(3))
    summary.to_csv("results_row_sweep_summary.csv")

    for metric in ("macro_f1", "accuracy"):
        table = runs_df.pivot_table(index=["model", "scale"], columns="n_rows",
                                    values=metric, aggfunc="mean").round(3)
        print(f"--- mean test {metric} over {len(SEEDS)} seeds "
              f"(rows = model and scale, columns = training rows) ---")
        print(table.to_string(), "\n")
    print(f"(reference: MLP on all {len(Xall)} rows: accuracy={full_acc:.3f} macroF1={full_f1:.3f})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for (model, scale), d in runs_df.groupby(["model", "scale"]):
            m = d.groupby("n_rows")["macro_f1"].mean()
            style = {"mlp": "k--", "angle-QNN": "-", "zz-QNN": ":"}[model]
            ax.plot(m.index, m.values, style, marker="o",
                    label="mlp" if model == "mlp" else f"{model} scale {scale}")
        ax.axhline(full_f1, color="gray", lw=0.8)
        ax.set_xlabel("training rows"); ax.set_ylabel("mean test macro F1")
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig("row_sweep.png", dpi=150)
        print("Saved row_sweep.png")
    except ImportError:
        pass
    print("Saved results_row_sweep_runs.csv and results_row_sweep_summary.csv")


if __name__ == "__main__":
    main()