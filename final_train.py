"""
final_train.py - train the final chain + CNOT VQC on the simulator, then stress-test it.

Needs Qubit_Pattern_Comparison_VQC_v2.py in the same folder (circuit, exact gradients, loss).
Run:   python final_train.py            (full run)
       python final_train.py --quick    (tiny settings, a minute or two, to check everything works)

What it does
  1. Splits the training CSV into  train / validation / FRESH test  (stratified). The fresh test is
     never used for any choice. Your original test CSV is also scored, but it was already used to
     pick chain/cnot, so treat it as a continuity check only.
  2. Trains N_RESTARTS random starts in parallel on the train split (L-BFGS-B, exact gradients),
     keeps the one with the lowest validation loss, then refits it on train+validation.
  3. Reports accuracy / macro F1 (+ bootstrap 95% CI) on the fresh test, next to a no-link VQC
     and a classical MLP trained on the same rows.
  4. Noise study (all in numpy, exact density matrices): depolarizing gate error, readout error and
     finite shots, with and without refitting the classical head on the noisy features.
  5. Saves everything to OUT_DIR, including the trained parameters (model_<topology>_<gate>.npz).

Circuit (qubit 0 = least significant bit, as in Qiskit), for the saved model:
  for block r = 0 .. ANSATZ_REPS:
      for every qubit q:   RY(a[r,q] * x_q + theta[r,q]);  then RZ(rz[r,q]) except in the last block
      if r < ANSATZ_REPS:  CNOT(0,1), CNOT(1,2), CNOT(2,3)   (control = first qubit of the pair)
  measure all qubits -> 2^n probabilities p -> logits = (2^n * p) @ W + bias  -> argmax
  x = MinMax-scaled features in [0, pi] using the saved per-feature lo/hi.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

try:
    import Qubit_Pattern_Comparison_VQC_v2 as m
except ImportError:
    sys.exit("Put Qubit_Pattern_Comparison_VQC_v2.py in the same folder as final_train.py")

# ----------------------------------------------------------------------------- settings
TRAIN_FILE = "cbc_train_lda.csv"
TEST_FILE = "cbc_test_lda.csv"          # your ORIGINAL test set (already used for model selection)
LABEL_COL = "diagnosis"
OUT_DIR = "final_model"

TOPOLOGY, GATE = "chain", "cnot"
READOUT = "probs"                       # "probs" or "z"
FULL_ROTATIONS = True
ANSATZ_REPS = 2
L2, LABEL_SMOOTHING, WEIGHT_POWER = 1e-3, 0.05, 0.5

VAL_FRAC = 0.15
FRESH_TEST_FRAC = 0.15
SPLIT_SEED = 0
N_RESTARTS = 10
MAXITER = 1000                          # L-BFGS iterations per restart
REFIT_ITER = 200                        # refit of the winner on train+validation
TRAIN_NO_LINK_BASELINE = True           # also train a model with no linking gates, same recipe
BOOTSTRAP = 1000

NOISE_LEVELS = (0.0, 0.005, 0.01, 0.02)  # two-qubit depolarizing probability per CNOT
SINGLE_QUBIT_RATIO = 0.1                 # single-qubit error = ratio * two-qubit error
READOUT_ERROR = 0.01                     # symmetric bit-flip prob per qubit (used when noise > 0)
SHOT_LEVELS = (None, 16384, 4096, 1024)  # None = exact probabilities
N_WORKERS = max(1, min(N_RESTARTS, (os.cpu_count() or 2) - 1))


def apply_cfg(cfg):
    """Set the v2 module's globals (must also run inside worker processes)."""
    m.READOUT, m.FULL_ROTATIONS, m.ANSATZ_REPS = cfg["readout"], cfg["full_rotations"], cfg["reps"]
    m.L2, m.LABEL_SMOOTHING, m.WEIGHT_POWER = cfg["l2"], cfg["smoothing"], cfg["weight_power"]


# ----------------------------------------------------------------------- data handling
def scale_features(X, lo, hi):
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    return (X - lo) / span * m.SCALE


def strat_split(idx, y, frac, seed):
    try:
        return train_test_split(idx, test_size=frac, random_state=seed, stratify=y[idx])
    except ValueError:      # a class too rare to stratify
        return train_test_split(idx, test_size=frac, random_state=seed)


def metrics(y_true, y_pred):
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")


def bootstrap_f1(y_true, y_pred, B, seed=0):
    rng = np.random.default_rng(seed)
    N = len(y_true)
    vals = [f1_score(y_true[i], y_pred[i], average="macro") for i in rng.integers(0, N, (B, N))]
    return np.percentile(vals, [2.5, 97.5])


# -------------------------------------------------------------------------- training
def restart_worker(args):
    cfg, topology, gate, C, n, Xs, Q, sw, Xv, Qv, swv, seed = args
    apply_cfg(cfg)
    model = m.Model(topology, gate, n, C)
    w0 = model.init(np.random.default_rng(seed))
    res = m.minimize(m.loss_grad, w0, args=(model, Xs, Q, sw), jac=True, method="L-BFGS-B",
                     options={"maxiter": cfg["maxiter"]})
    return res.x, res.fun, m.data_loss(res.x, model, Xv, Qv, swv), res.nit


def train_variant(cfg, topology, gate, Xtr, ytr, Xva, yva, pool):
    """Best-of-N restarts by validation loss, then refit on train+validation."""
    n, C = Xtr.shape[1], cfg["C"]
    Qtr, Qva = m.smooth_targets(ytr, C), m.smooth_targets(yva, C)
    counts = np.bincount(ytr, minlength=C)
    swtr, swva = m.class_sample_weights(counts, ytr), m.class_sample_weights(counts, yva)
    jobs = [(cfg, topology, gate, C, n, Xtr, Qtr, swtr, Xva, Qva, swva, 1000 + i)
            for i in range(cfg["restarts"])]
    outs = list(pool.map(restart_worker, jobs)) if pool else [restart_worker(j) for j in jobs]

    print(f"  restarts for {topology}/{gate}:  train_loss  val_loss  iterations")
    for i, (_, tl, vl, nit) in enumerate(outs):
        flag = "   <- hit MAXITER" if nit >= cfg["maxiter"] else ""
        print(f"    #{i:<2d}            {tl:9.4f} {vl:9.4f} {nit:9d}{flag}")
    k = int(np.argmin([o[2] for o in outs]))
    print(f"  best restart: #{k}")

    apply_cfg(cfg)
    model = m.Model(topology, gate, n, C)
    w = outs[k][0]
    val_loss_before = outs[k][2]
    if cfg["refit"] > 0:
        Xall, yall = np.vstack([Xtr, Xva]), np.concatenate([ytr, yva])
        sw = m.class_sample_weights(np.bincount(yall, minlength=C), yall)
        res = m.minimize(m.loss_grad, w, args=(model, Xall, m.smooth_targets(yall, C), sw), jac=True,
                         method="L-BFGS-B", options={"maxiter": cfg["refit"]})
        w = res.x
    return model, w, val_loss_before


# --------------------------------------------------------------------- noisy simulation
Y2 = np.array([[0, -1j], [1j, 0]])
PAULIS = {"I": m.I2.astype(complex), "X": m.PX.astype(complex), "Y": Y2, "Z": m.PZ.astype(complex)}


def pauli_ops(qubits, n):
    """All non-identity Pauli products on the given qubits, embedded as 2^n x 2^n matrices."""
    import itertools
    ops = []
    for combo in itertools.product("IXYZ", repeat=len(qubits)):
        if all(c == "I" for c in combo):
            continue
        ops.append(m.embed({q: PAULIS[c] for q, c in zip(qubits, combo)}, n))
    return ops


def depolarize(rho, ops, p):
    if p <= 0:
        return rho
    acc = sum(P @ rho @ P for P in ops)
    return (1 - p) * rho + (p / len(ops)) * acc


def noisy_probs(w, model, Xs, p2, p1, ro_err, chunk=400):
    """Basis-state probabilities with depolarizing gate noise (density matrices) and readout error.
    Supports fixed-gate models (cnot/cz) and the no-link model."""
    assert not model.trainable_links, "noise study is written for fixed gates (cnot / cz)"
    n, D, R = model.n, model.D, m.ANSATZ_REPS + 1
    p = model.unpack(w)
    pairs = m.link_pairs(model.topology, n) if model.links else []
    ops1 = [pauli_ops([q], n) for q in range(n)]
    ops2 = [pauli_ops(list(pr), n) for pr in pairs]
    out = []
    for s in range(0, len(Xs), chunk):
        X = Xs[s:s + chunk]
        N = len(X)
        rho = np.zeros((N, D, D), complex)
        rho[:, 0, 0] = 1.0
        for r in range(R):
            for q in range(n):
                t = p["a"][r, q] * X[:, q] + p["theta"][r, q]
                phi = p["rz"][r, q] if (m.FULL_ROTATIONS and r < m.ANSATZ_REPS) else None
                G, _, _ = m.gate_mats(t, phi)
                flat = rho.reshape(N, D * D)
                flat = m.apply1(flat, G, q + n, 2 * n)                  # ket index
                flat = m.apply1(flat, np.conj(G), q, 2 * n)             # bra index
                rho = depolarize(flat.reshape(N, D, D), ops1[q], p1)
            if r < m.ANSATZ_REPS:
                for L, ops in zip(model.links, ops2):
                    F = L.fixed
                    rho = depolarize(F @ rho @ F.T, ops, p2)
        out.append(np.real(np.diagonal(rho, axis1=1, axis2=2)))
    P = np.clip(np.vstack(out), 0, None)
    P /= P.sum(axis=1, keepdims=True)
    if ro_err > 0:
        M1 = np.array([[1 - ro_err, ro_err], [ro_err, 1 - ro_err]])
        M = np.ones((1, 1))
        for _ in range(n):
            M = np.kron(M, M1)
        P = P @ M.T
    return P


def features_from_probs(P, model, rng=None, shots=None):
    if shots is not None:
        P = P / P.sum(axis=1, keepdims=True)
        P = P.copy()
        P[:, -1] = np.clip(1 - P[:, :-1].sum(axis=1), 0, None)
        P = rng.multinomial(shots, P) / shots
    return model.D * P if m.READOUT == "probs" else P @ model.Zs


def refit_head(F, yi, C, W0, b0):
    """Refit only the classical head (W, bias) on given features; circuit parameters stay fixed."""
    nf = F.shape[1]
    Q = m.smooth_targets(yi, C)
    sw = m.class_sample_weights(np.bincount(yi, minlength=C), yi)
    tot = sw.sum()

    def f(v):
        W, b = v[:nf * C].reshape(nf, C), v[nf * C:]
        logp = m.log_softmax(F @ W + b)
        loss = -np.sum(sw[:, None] * Q * logp) / tot + m.L2 * np.sum(W ** 2)
        d = sw[:, None] * (np.exp(logp) - Q) / tot
        return loss, np.concatenate([(F.T @ d + 2 * m.L2 * W).ravel(), d.sum(axis=0)])

    res = minimize(f, np.concatenate([W0.ravel(), b0]), jac=True, method="L-BFGS-B",
                   options={"maxiter": 500})
    return res.x[:nf * C].reshape(nf, C), res.x[nf * C:]


def noise_study(cfg, model, w, Xtr, ytr, Xte, yte, classes, seed=0):
    apply_cfg(cfg)
    p = model.unpack(w)
    rng = np.random.default_rng(seed)

    # self-check: with no noise the density-matrix simulator must match the statevector one
    psi, _ = model.forward(p, Xte[:20])
    assert np.allclose(noisy_probs(w, model, Xte[:20], 0, 0, 0), np.abs(psi) ** 2, atol=1e-9), \
        "density-matrix simulator disagrees with the statevector simulator"

    rows = []
    for p2 in NOISE_LEVELS:
        p1, ro = p2 * SINGLE_QUBIT_RATIO, (READOUT_ERROR if p2 > 0 else 0.0)
        t0 = time.perf_counter()
        Pte = noisy_probs(w, model, Xte, p2, p1, ro)
        Ptr = noisy_probs(w, model, Xtr, p2, p1, ro)
        Wn, bn = refit_head(features_from_probs(Ptr, model), ytr, cfg["C"], p["W"], p["bias"])
        for shots in SHOT_LEVELS:
            F = features_from_probs(Pte, model, rng, shots)
            for tag, W, b in (("as_trained", p["W"], p["bias"]), ("head_refit", Wn, bn)):
                pred = classes[np.argmax(F @ W + b, axis=1)]
                acc, f1 = metrics(yte, pred)
                rows.append({"two_qubit_error": p2, "readout_error": ro,
                             "shots": "exact" if shots is None else shots,
                             "head": tag, "accuracy": acc, "macro_f1": f1})
        print(f"  noise level {p2:.3f} done ({time.perf_counter() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ save / load
def save_model(path, cfg, topology, gate, w, classes, lo, hi, n):
    np.savez(path, w=w, classes=classes.astype(str), lo=lo, hi=hi, n=n, topology=topology, gate=gate,
             readout=cfg["readout"], full_rotations=cfg["full_rotations"], reps=cfg["reps"],
             scale=m.SCALE)


def predict_from_file(path, X):
    """Load a saved model and classify raw (unscaled) feature rows. Returns (labels, probabilities)."""
    d = np.load(path)
    apply_cfg({"readout": str(d["readout"]), "full_rotations": bool(d["full_rotations"]),
               "reps": int(d["reps"]), "l2": L2, "smoothing": LABEL_SMOOTHING,
               "weight_power": WEIGHT_POWER})
    classes = d["classes"]
    model = m.Model(str(d["topology"]), str(d["gate"]), int(d["n"]), len(classes))
    logits = m.predict_logits(d["w"], model, scale_features(X, d["lo"], d["hi"]))
    P = np.exp(m.log_softmax(logits))
    return classes[np.argmax(logits, axis=1)], P


# ------------------------------------------------------------------------- main
def main(quick=False):
    global NOISE_LEVELS, SHOT_LEVELS, BOOTSTRAP
    restarts, maxiter, refit = N_RESTARTS, MAXITER, REFIT_ITER
    if quick:
        restarts, maxiter, refit = 2, 60, 20
        NOISE_LEVELS, SHOT_LEVELS, BOOTSTRAP = (0.0, 0.01), (None, 4096), 100
    os.makedirs(OUT_DIR, exist_ok=True)

    tr, te = pd.read_csv(TRAIN_FILE), pd.read_csv(TEST_FILE)
    feats = [c for c in tr.columns if c != LABEL_COL]
    Xall, yall = tr[feats].values, tr[LABEL_COL].astype(str).values
    Xold, yold = te[feats].values, te[LABEL_COL].astype(str).values
    classes = np.unique(yall)
    yi_all = np.searchsorted(classes, yall)

    idx = np.arange(len(Xall))
    rest, fresh = strat_split(idx, yall, FRESH_TEST_FRAC, SPLIT_SEED)
    train, val = strat_split(rest, yall, VAL_FRAC / (1 - FRESH_TEST_FRAC), SPLIT_SEED + 1)
    print(f"{len(train)} train / {len(val)} validation / {len(fresh)} fresh-test rows; "
          f"{len(Xold)} rows in the original test file; {len(feats)} qubits, {len(classes)} classes")

    fit_rows = np.concatenate([train, val])
    lo, hi = Xall[fit_rows].min(axis=0), Xall[fit_rows].max(axis=0)
    S = lambda X: scale_features(X, lo, hi)
    cfg = {"readout": READOUT, "full_rotations": FULL_ROTATIONS, "reps": ANSATZ_REPS, "l2": L2,
           "smoothing": LABEL_SMOOTHING, "weight_power": WEIGHT_POWER, "C": len(classes),
           "restarts": restarts, "maxiter": maxiter, "refit": refit}
    apply_cfg(cfg)

    variants = [(TOPOLOGY, GATE)] + ([("none", "none")] if TRAIN_NO_LINK_BASELINE else [])
    results, models = [], {}
    pool = ProcessPoolExecutor(max_workers=N_WORKERS) if N_WORKERS > 1 else None
    try:
        for topology, gate in variants:
            print(f"\n=== training {topology}/{gate} ===", flush=True)
            t0 = time.perf_counter()
            model, w, _ = train_variant(cfg, topology, gate, S(Xall[train]), yi_all[train],
                                        S(Xall[val]), yi_all[val], pool)
            print(f"  trained in {time.perf_counter() - t0:.0f}s, {model.n_params} parameters")
            models[(topology, gate)] = (model, w)
            for name, X, y in (("fresh_test", Xall[fresh], yall[fresh]), ("original_test", Xold, yold)):
                pred = classes[np.argmax(m.predict_logits(w, model, S(X)), axis=1)]
                acc, f1 = metrics(y, pred)
                lo_ci, hi_ci = bootstrap_f1(y, pred, BOOTSTRAP)
                results.append({"model": f"vqc {topology}/{gate}", "set": name, "accuracy": acc,
                                "macro_f1": f1, "f1_ci_low": lo_ci, "f1_ci_high": hi_ci})
                if name == "fresh_test" and (topology, gate) == (TOPOLOGY, GATE):
                    print("\n  per-class report, fresh test:\n" + classification_report(
                        y, pred, zero_division=0))
            save_model(os.path.join(OUT_DIR, f"model_{topology}_{gate}.npz"), cfg, topology, gate,
                       w, classes, lo, hi, len(feats))
    finally:
        if pool:
            pool.shutdown()

    # classical reference on the same rows
    ss = StandardScaler().fit(Xall[fit_rows])
    mlp = MLPClassifier(hidden_layer_sizes=m.HIDDEN, max_iter=2000, random_state=0)
    mlp.fit(ss.transform(Xall[fit_rows]), yall[fit_rows])
    for name, X, y in (("fresh_test", Xall[fresh], yall[fresh]), ("original_test", Xold, yold)):
        pred = mlp.predict(ss.transform(X))
        acc, f1 = metrics(y, pred)
        lo_ci, hi_ci = bootstrap_f1(y, pred, BOOTSTRAP)
        results.append({"model": "mlp", "set": name, "accuracy": acc, "macro_f1": f1,
                        "f1_ci_low": lo_ci, "f1_ci_high": hi_ci})

    res_df = pd.DataFrame(results).round(3)
    res_df.to_csv(os.path.join(OUT_DIR, "final_metrics.csv"), index=False)
    print("\n=== noiseless results (CI = bootstrap 95% on macro F1) ===")
    print(res_df.to_string(index=False))
    print("\nNote: 'original_test' was already used to choose chain/cnot, so it is optimistic.")

    print("\n=== noise study (chain/cnot model) ===", flush=True)
    model, w = models[(TOPOLOGY, GATE)]
    ns = noise_study(cfg, model, w, S(Xall[train]), yi_all[train], S(Xall[fresh]), yall[fresh], classes)
    ns.round(4).to_csv(os.path.join(OUT_DIR, "noise_study.csv"), index=False)
    for tag in ("as_trained", "head_refit"):
        t = ns[ns["head"] == tag].pivot_table(index=["two_qubit_error", "readout_error"],
                                              columns="shots", values="macro_f1", sort=False).round(3)
        print(f"\nmacro F1 on fresh test, head = {tag}  (rows: gate error, readout error; cols: shots)")
        print(t.to_string())
    json.dump({**cfg, "topology": TOPOLOGY, "gate": GATE, "classes": classes.tolist(),
               "noise_levels": NOISE_LEVELS, "readout_error": READOUT_ERROR},
              open(os.path.join(OUT_DIR, "config.json"), "w"), indent=1)
    print(f"\nSaved everything to {OUT_DIR}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny settings to check the pipeline")
    main(ap.parse_args().quick)
