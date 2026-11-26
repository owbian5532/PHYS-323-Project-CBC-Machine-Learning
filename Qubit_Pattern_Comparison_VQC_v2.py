
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import reduce
from itertools import combinations
from scipy.optimize import minimize
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

TRAIN_FILE = "cbc_train_lda.csv"
TEST_FILE = "cbc_test_lda.csv"
LABEL_COL = "diagnosis"

N_TRAIN_ROWS = 500
SEEDS = tuple(range(10))
SCALE = np.pi                # features scaled to [0, SCALE]; a[r,q] then rescales them per layer
ANSATZ_REPS = 2              # linking layers; there are ANSATZ_REPS + 1 encoding blocks
HIDDEN = (16,)               # classical MLP reference

# ---- architecture switches
READOUT = "probs"            # "probs" (2^n basis probabilities) or "z" (n <Z> values, as in v1)
FULL_ROTATIONS = True        # add trainable RZ after each RY

# ---- optimizer (L-BFGS-B, exact gradients)
MAXITER = 300                # L-BFGS iterations per restart
N_RESTARTS = 3               # random starts; the one with the lowest VALIDATION loss is kept
VAL_FRAC = 0.2               # share of the sampled rows held out to choose between restarts
REFIT_ITER = 100             # extra iterations refitting the winner on ALL sampled rows (0 = skip)

# ---- loss
WEIGHT_POWER = 0.5           # class weight = count^-WEIGHT_POWER (1.0 = fully 'balanced', 0 = none)
LABEL_SMOOTHING = 0.05
L2 = 1e-3                    # penalty on the head weights W only

TOPOLOGIES = ["chain", "ring", "full", "star"]
GATES = ["cnot", "cz", "cry", "rzz"]
TRAINABLE_GATES = {"cry", "rzz"}
INCLUDE_NO_LINK_CONTROL = True
N_WORKERS = max(1, min(len(SEEDS), (os.cpu_count() or 2) - 1))


# ------------------------------------------------------------ linking patterns
def link_pairs(topology, n):
    chain = [(i, i + 1) for i in range(n - 1)]
    if topology == "chain":
        return chain
    if topology == "ring":
        return chain + [(n - 1, 0)]
    if topology == "full":
        return list(combinations(range(n), 2))
    if topology == "star":
        return [(0, j) for j in range(1, n)]
    raise ValueError(topology)


# -------------------------------------------------------------- matrix building
I2 = np.eye(2)
PX = np.array([[0.0, 1.0], [1.0, 0.0]])
PZ = np.diag([1.0, -1.0])
P0 = np.diag([1.0, 0.0])
P1 = np.diag([0.0, 1.0])
J = np.array([[0.0, -1.0], [1.0, 0.0]])    # RY(t) = cos(t/2) I + sin(t/2) J


def embed(ops, n):
    return reduce(np.kron, [ops.get(q, I2) for q in reversed(range(n))])


class Link:
    """One two-qubit linking gate between qubits a and b, acting on a batch of row-state-vectors."""

    def __init__(self, gate, a, b, n):
        self.gate = gate
        self.trainable = gate in TRAINABLE_GATES
        if gate == "cnot":
            self.fixed = embed({a: P0}, n) + embed({a: P1, b: PX}, n)
        elif gate == "cz":
            self.fixed = embed({a: P0}, n) + embed({a: P1, b: PZ}, n)
        elif gate == "cry":      # controlled-RY(t) = base + cos(t/2)*c_part + sin(t/2)*s_part
            self.base = embed({a: P0}, n)
            self.c_part = embed({a: P1}, n)
            self.s_part = embed({a: P1, b: J}, n)
        elif gate == "rzz":      # exp(-i t/2 Z_a Z_b): diagonal
            self.zz = np.diag(embed({a: PZ, b: PZ}, n)).copy()
        else:
            raise ValueError(gate)

    def forward(self, t, psi):
        if self.gate == "cry":
            c, s = np.cos(t / 2), np.sin(t / 2)
            U = self.base + c * self.c_part + s * self.s_part
            return psi @ U.T
        d = np.cos(t / 2) - 1j * np.sin(t / 2) * self.zz
        return d[None, :] * psi

    def backward(self, t, psi_in, g):
        """Returns (gradient w.r.t. the input state, gradient w.r.t. t) given dL/d(conj psi_out)=g."""
        if self.gate == "cry":
            c, s = np.cos(t / 2), np.sin(t / 2)
            U = self.base + c * self.c_part + s * self.s_part
            dU = -0.5 * s * self.c_part + 0.5 * c * self.s_part
            dpsi = psi_in @ dU.T
            return g @ U, 2.0 * np.real(np.sum(np.conj(g) * dpsi))
        d = np.cos(t / 2) - 1j * np.sin(t / 2) * self.zz
        dd = -0.5 * np.sin(t / 2) - 0.5j * np.cos(t / 2) * self.zz
        dpsi = dd[None, :] * psi_in
        return np.conj(d)[None, :] * g, 2.0 * np.real(np.sum(np.conj(g) * dpsi))


# ---------------------------------------------- batched single-qubit gates
def apply1(psi, G, q, n):
    """Apply per-sample 2x2 matrices G (N,2,2) to qubit q of a batch of states psi (N, 2^n)."""
    N = psi.shape[0]
    v = psi.reshape(N, 2 ** (n - q - 1), 2, 2 ** q)
    p0, p1 = v[:, :, 0, :], v[:, :, 1, :]

    def g(i, j):
        return G[:, i, j][:, None, None]

    out = np.empty_like(v)
    out[:, :, 0, :] = g(0, 0) * p0 + g(0, 1) * p1
    out[:, :, 1, :] = g(1, 0) * p0 + g(1, 1) * p1
    return out.reshape(N, -1)


def gate_mats(t, phi):
    """G = RZ(phi) RY(t) per sample (t is an array), plus dG/dt and dG/dphi (None if phi is None)."""
    c, s = np.cos(t / 2), np.sin(t / 2)
    ph = 0.0 if phi is None else phi
    em, ep = np.exp(-0.5j * ph), np.exp(0.5j * ph)
    N = len(t)
    G = np.zeros((N, 2, 2), complex)
    G[:, 0, 0], G[:, 0, 1], G[:, 1, 0], G[:, 1, 1] = em * c, -em * s, ep * s, ep * c
    dT = np.zeros((N, 2, 2), complex)
    dT[:, 0, 0], dT[:, 0, 1], dT[:, 1, 0], dT[:, 1, 1] = -0.5 * em * s, -0.5 * em * c, 0.5 * ep * c, -0.5 * ep * s
    dP = None
    if phi is not None:
        dP = np.zeros((N, 2, 2), complex)
        dP[:, 0, 0], dP[:, 0, 1] = -0.5j * em * c, 0.5j * em * s
        dP[:, 1, 0], dP[:, 1, 1] = 0.5j * ep * s, 0.5j * ep * c
    return G, dT, dP


def z_signs(n):
    """(2^n x n): entry [k, j] = +1 if qubit j is 0 in basis state k, else -1."""
    return np.array([[1.0 - 2 * ((k >> j) & 1) for j in range(n)] for k in range(2 ** n)])


class Model:
    """Structure of one VQC variant plus its forward pass and exact gradient."""

    def __init__(self, topology, gate, n, C):
        self.n, self.C, self.D = n, C, 2 ** n
        self.topology, self.gate = topology, gate
        self.links = [] if topology == "none" else [Link(gate, a, b, n) for a, b in link_pairs(topology, n)]
        self.trainable_links = bool(self.links) and self.links[0].trainable
        R = ANSATZ_REPS + 1
        self.shapes = {
            "a": (R, n),
            "theta": (R, n),
            "rz": (ANSATZ_REPS if FULL_ROTATIONS else 0, n),
            "phi": (ANSATZ_REPS, len(self.links)) if self.trainable_links else (0, 0),
        }
        self.Zs = z_signs(n)
        self.n_feat = self.D if READOUT == "probs" else n
        self.shapes["W"] = (self.n_feat, C)
        self.shapes["bias"] = (C,)
        self.sizes = {k: int(np.prod(v)) for k, v in self.shapes.items()}
        self.n_params = sum(self.sizes.values())
        self.E = None
        if self.links and not self.trainable_links:
            E = np.eye(self.D)
            for L in self.links:
                E = L.fixed @ E
            self.E = E

    def unpack(self, w):
        out, i = {}, 0
        for k in ("a", "theta", "rz", "phi", "W", "bias"):
            out[k] = w[i:i + self.sizes[k]].reshape(self.shapes[k])
            i += self.sizes[k]
        return out

    def init(self, rng):
        w = np.zeros(self.n_params)
        p = self.unpack(w)                      # views into w
        p["a"][:] = 1.0 + rng.normal(0, 0.1, p["a"].shape)
        p["theta"][:] = rng.uniform(0, 2 * np.pi, p["theta"].shape)
        p["rz"][:] = rng.uniform(0, 2 * np.pi, p["rz"].shape)
        p["phi"][:] = rng.uniform(0, 2 * np.pi, p["phi"].shape)
        p["W"][:] = rng.normal(0, 1.0, p["W"].shape)
        return w

    # ---- circuit
    def forward(self, p, X):
        n, N = self.n, len(X)
        psi = np.zeros((N, self.D), complex)
        psi[:, 0] = 1.0
        cache = []
        for r in range(ANSATZ_REPS + 1):
            for q in range(n):
                t = p["a"][r, q] * X[:, q] + p["theta"][r, q]
                phi = p["rz"][r, q] if (FULL_ROTATIONS and r < ANSATZ_REPS) else None
                G, _, _ = gate_mats(t, phi)
                cache.append(("gate", r, q, psi, t, phi))
                psi = apply1(psi, G, q, n)
            if r < ANSATZ_REPS and self.links:
                if self.E is not None:
                    cache.append(("E", psi))
                    psi = psi @ self.E.T
                else:
                    for k, L in enumerate(self.links):
                        cache.append(("link", r, k, psi))
                        psi = L.forward(p["phi"][r, k], psi)
        return psi, cache

    def backward(self, p, gp, cache, g, X):
        """Reverse-mode through the circuit. g = dL/d(conj psi_final); fills gradient views gp."""
        n = self.n
        for item in reversed(cache):
            kind = item[0]
            if kind == "gate":
                _, r, q, psi_in, t, phi = item
                G, dT, dP = gate_mats(t, phi)
                rt = 2.0 * np.real(np.sum(np.conj(g) * apply1(psi_in, dT, q, n), axis=1))
                gp["theta"][r, q] += rt.sum()
                gp["a"][r, q] += (rt * X[:, q]).sum()
                if dP is not None:
                    gp["rz"][r, q] += 2.0 * np.real(np.sum(np.conj(g) * apply1(psi_in, dP, q, n)))
                g = apply1(g, np.conj(G).transpose(0, 2, 1), q, n)
            elif kind == "E":
                g = g @ self.E                      # E is real, so conj(E) = E
            else:
                _, r, k, psi_in = item
                g, gt = self.links[k].backward(p["phi"][r, k], psi_in, g)
                gp["phi"][r, k] += gt

    def features(self, psi):
        prob = np.abs(psi) ** 2
        return self.D * prob if READOUT == "probs" else prob @ self.Zs


# ------------------------------------------------------------------------- loss
def log_softmax(L):
    L = L - L.max(axis=1, keepdims=True)
    return L - np.log(np.exp(L).sum(axis=1, keepdims=True))


def class_sample_weights(counts, yi):
    """Per-class weight count^-WEIGHT_POWER, normalised so the average sample weight is 1."""
    w = np.maximum(counts, 1).astype(float) ** (-WEIGHT_POWER)
    w *= counts.sum() / max((w * counts).sum(), 1e-12)
    return w[yi]


def smooth_targets(yi, C):
    Q = np.full((len(yi), C), LABEL_SMOOTHING / C)
    Q[np.arange(len(yi)), yi] += 1.0 - LABEL_SMOOTHING
    return Q


def loss_grad(w, model, X, Q, sw, l2=L2):
    """Weighted smoothed cross-entropy + L2(W), and its exact gradient w.r.t. all parameters."""
    p = model.unpack(w)
    psi, cache = model.forward(p, X)
    F = model.features(psi)
    logp = log_softmax(F @ p["W"] + p["bias"])
    tot = sw.sum()
    loss = -np.sum(sw[:, None] * Q * logp) / tot + l2 * np.sum(p["W"] ** 2)

    dlog = sw[:, None] * (np.exp(logp) - Q) / tot           # dL/dlogits
    gw = np.zeros_like(w)
    gp = model.unpack(gw)
    gp["W"][:] = F.T @ dlog + 2 * l2 * p["W"]
    gp["bias"][:] = dlog.sum(axis=0)
    dF = dlog @ p["W"].T
    dprob = model.D * dF if READOUT == "probs" else dF @ model.Zs.T
    model.backward(p, gp, cache, dprob * psi, X)            # dL/d(conj psi) = dL/dprob * psi
    return loss, gw


def data_loss(w, model, X, Q, sw):
    return loss_grad(w, model, X, Q, sw, l2=0.0)[0]


def predict_logits(w, model, X):
    p = model.unpack(w)
    psi, _ = model.forward(p, X)
    return model.features(psi) @ p["W"] + p["bias"]


def fit_predict_vqc(Xtr, ytr, Xte, model_spec, seed):
    """Scale on TRAIN only, train with L-BFGS-B (best of N_RESTARTS by validation loss, then refit
    on all rows), predict Xte. Returns (predictions, final training loss, n_params)."""
    topology, gate = model_spec
    sc = MinMaxScaler((0, SCALE)).fit(Xtr)
    Xs, Xt = sc.transform(Xtr), sc.transform(Xte)

    classes = np.unique(ytr)
    C, n = len(classes), Xtr.shape[1]
    yi = np.searchsorted(classes, ytr)
    Q = smooth_targets(yi, C)
    model = Model(topology, gate, n, C)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(yi))
    n_val = int(VAL_FRAC * len(yi)) if N_RESTARTS > 1 else 0
    va, tr = perm[:n_val], perm[n_val:]
    sw_tr = class_sample_weights(np.bincount(yi[tr], minlength=C), yi[tr])
    sw_va = class_sample_weights(np.bincount(yi[tr], minlength=C), yi[va]) if n_val else None

    best, best_val = None, np.inf
    for _ in range(N_RESTARTS):
        res = minimize(loss_grad, model.init(rng), args=(model, Xs[tr], Q[tr], sw_tr), jac=True,
                       method="L-BFGS-B", options={"maxiter": MAXITER})
        val = data_loss(res.x, model, Xs[va], Q[va], sw_va) if n_val else res.fun
        if val < best_val:
            best, best_val = res, val

    if REFIT_ITER > 0 and n_val:
        sw_all = class_sample_weights(np.bincount(yi, minlength=C), yi)
        best = minimize(loss_grad, best.x, args=(model, Xs, Q, sw_all), jac=True,
                        method="L-BFGS-B", options={"maxiter": REFIT_ITER})
    return classes[np.argmax(predict_logits(best.x, model, Xt), axis=1)], best.fun, model.n_params


def fit_predict_mlp(Xtr, ytr, Xte, seed):
    ss = StandardScaler().fit(Xtr)
    clf = MLPClassifier(hidden_layer_sizes=HIDDEN, max_iter=2000, random_state=seed)
    return clf.fit(ss.transform(Xtr), ytr).predict(ss.transform(Xte))


# ------------------------------------------------------------------ evaluation
def model_specs():
    specs = []
    if INCLUDE_NO_LINK_CONTROL:
        specs.append(("none", "none"))
    specs += [(t, g) for t in TOPOLOGIES for g in GATES]
    return specs


def score(y_true, y_pred):
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")


def run_seed(seed, Xall, yall, Xte, yte):
    order = np.random.default_rng(seed).permutation(len(Xall))[:N_TRAIN_ROWS]
    X, y = Xall[order], yall[order]
    n_cls = len(set(y))
    rows = []

    def record(model, topology, gate, n_params, loss, y_pred):
        acc, f1 = score(yte, y_pred)
        rows.append({"model": model, "topology": topology, "gate": gate, "n_params": n_params,
                     "seed": seed, "classes_in_sample": n_cls, "train_loss": loss,
                     "accuracy": acc, "macro_f1": f1})

    record("mlp", "-", "-", np.nan, np.nan, fit_predict_mlp(X, y, Xte, seed))
    for topology, gate in model_specs():
        pred, loss, n_params = fit_predict_vqc(X, y, Xte, (topology, gate), seed)
        record("vqc", topology, gate, n_params, loss, pred)
    return seed, rows, f"seed {seed} finished ({n_cls} of {len(set(yall))} diagnoses in its sample)"


def main():
    tr = pd.read_csv(TRAIN_FILE)
    te = pd.read_csv(TEST_FILE)
    feats = [c for c in tr.columns if c != LABEL_COL]
    Xall, yall = tr[feats].values, tr[LABEL_COL].astype(str).values
    Xte, yte = te[feats].values, te[LABEL_COL].astype(str).values
    print(f"{len(Xall)} train rows available, {N_TRAIN_ROWS} sampled per seed; {len(Xte)} test rows, "
          f"{Xall.shape[1]} qubits, {len(SEEDS)} seeds on {N_WORKERS} worker process(es), "
          f"{len(model_specs())} VQC variants, readout={READOUT}, full_rotations={FULL_ROTATIONS}",
          flush=True)

    full_acc, full_f1 = score(yte, fit_predict_mlp(Xall, yall, Xte, 0))
    print(f"Reference, MLP on all {len(Xall)} rows: accuracy={full_acc:.3f} macroF1={full_f1:.3f}\n",
          flush=True)

    runs, t0 = [], time.perf_counter()
    if N_WORKERS > 1:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = [pool.submit(run_seed, s, Xall, yall, Xte, yte) for s in SEEDS]
            for fut in as_completed(futures):
                seed, rows, log = fut.result()
                runs.extend(rows)
                print(f"[{time.perf_counter() - t0:5.0f}s] {log}", flush=True)
    else:
        for s in SEEDS:
            seed, rows, log = run_seed(s, Xall, yall, Xte, yte)
            runs.extend(rows)
            print(f"[{time.perf_counter() - t0:5.0f}s] {log}", flush=True)

    df = pd.DataFrame(runs)
    df.to_csv("results_qubit_patterns_runs.csv", index=False)
    summary = (df.groupby(["model", "topology", "gate", "n_params"], dropna=False)
               [["accuracy", "macro_f1", "train_loss"]].agg(["mean", "std"]).round(3))
    summary.to_csv("results_qubit_patterns_summary.csv")

    vq = df[df.model == "vqc"]
    for metric in ("macro_f1", "accuracy", "train_loss"):
        table = vq.pivot_table(index="topology", columns="gate", values=metric,
                               aggfunc="mean").round(3)
        print(f"\n--- mean {metric} over {len(SEEDS)} seeds "
              f"(rows = pattern, columns = linking gate) ---")
        print(table.to_string())
    m = df[df.model == "mlp"]
    print(f"\n(MLP on the same {N_TRAIN_ROWS} rows: accuracy={m.accuracy.mean():.3f} "
          f"macroF1={m.macro_f1.mean():.3f};  MLP on all rows: accuracy={full_acc:.3f} "
          f"macroF1={full_f1:.3f})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        g = vq.groupby(["topology", "gate"])["macro_f1"].agg(["mean", "std"]).reset_index()
        labels = [f"{t}\n{ga}" if t != "none" else "none" for t, ga in zip(g.topology, g.gate)]
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(range(len(g)), g["mean"], yerr=g["std"], capsize=3)
        ax.axhline(m.macro_f1.mean(), color="k", ls="--", lw=0.8, label=f"MLP, same {N_TRAIN_ROWS} rows")
        ax.set_xticks(range(len(g))); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("mean test macro F1 (error bar = std over seeds)")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("qubit_patterns.png", dpi=150)
        print("Saved qubit_patterns.png")
    except ImportError:
        pass
    print("Saved results_qubit_patterns_runs.csv and results_qubit_patterns_summary.csv")


if __name__ == "__main__":
    main()
