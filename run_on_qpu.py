"""
run_on_qpu.py - take the trained chain/CNOT model and run it on a qBraid simulator or a real quantum chip.

Put this file in the SAME folder as final_train.py, Qubit_Pattern_Comparison_VQC_v2.py,
cbc_test_lda.csv and the final_model folder. Then run these in order (in the qBraid terminal):

    python run_on_qpu.py --list       (1) shows the devices your account can use
    python run_on_qpu.py --check      (2) checks the rebuilt circuit matches your trained model (no hardware)
    python run_on_qpu.py --run        (3) sends patients to the device named in DEVICE_ID below

Only ONE thing needs editing: DEVICE_ID.
"""
import argparse
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.metrics import accuracy_score, f1_score

import final_train as ft
import Qubit_Pattern_Comparison_VQC_v2 as m

# ============================ EDIT THESE ============================
DEVICE_ID = "PASTE_DEVICE_ID_HERE"   # copy a name from the --list output
N_PATIENTS = 1                       # start with 1 on a real chip; raise it once you see the cost
SHOTS = 1000                         # how many times each circuit is repeated
# ====================================================================

MODEL_FILE = "final_model/model_chain_cnot.npz"
TEST_FILE = "cbc_test_lda.csv"
LABEL_COL = "diagnosis"

# ---------------------------------------------------------------- load the trained model
d = np.load(MODEL_FILE)
classes = d["classes"]
n = int(d["n"])
reps = int(d["reps"])
ft.apply_cfg({"readout": str(d["readout"]), "full_rotations": bool(d["full_rotations"]),
              "reps": reps, "l2": ft.L2, "smoothing": ft.LABEL_SMOOTHING,
              "weight_power": ft.WEIGHT_POWER})
model = m.Model(str(d["topology"]), str(d["gate"]), n, len(classes))
p = model.unpack(d["w"])


def load_patients(count):
    """Pick `count` random patients from the test file (fixed seed so reruns use the same ones)."""
    te = pd.read_csv(TEST_FILE)
    feats = [c for c in te.columns if c != LABEL_COL]
    pick = np.random.default_rng(0).permutation(len(te))[:count]
    X = ft.scale_features(te[feats].values[pick], d["lo"], d["hi"])
    y = te[LABEL_COL].astype(str).values[pick]
    return X, y


def build(x, measure=True):
    """The trained circuit for one patient x (4 scaled numbers)."""
    qc = QuantumCircuit(n)
    for r in range(reps + 1):
        for q in range(n):
            qc.ry(float(p["a"][r, q] * x[q] + p["theta"][r, q]), q)
            if r < reps:
                qc.rz(float(p["rz"][r, q]), q)
        if r < reps:
            for q in range(n - 1):
                qc.cx(q, q + 1)
    if measure:
        qc.measure_all()
    return qc


def predict_from_probs(P):
    """16 probabilities -> diagnosis, using the trained scoring step."""
    logits = (model.D * P) @ p["W"] + p["bias"]
    return classes[np.argmax(logits)]


def predict_from_counts(counts):
    P = np.zeros(2 ** n)
    total = sum(counts.values())
    for bits, c in counts.items():
        P[int(bits.replace(" ", ""), 2)] += c / total
    return predict_from_probs(P)


def get_counts(result):
    """Pull the {bitstring: count} table out of a qBraid result (handles a couple of formats)."""
    for getter in (lambda: result.data.get_counts(), lambda: result.measurement_counts()):
        try:
            return dict(getter())
        except Exception:
            pass
    raise RuntimeError(f"Could not read counts. Here is the raw result, send it to Claude:\n{result}")


def score(y_true, y_pred):
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")


# ---------------------------------------------------------------- the three modes
def do_list():
    from qbraid.runtime import QbraidProvider
    for dev in QbraidProvider().get_devices():
        print(dev)


def do_check():
    X, y = load_patients(20)
    exact = np.abs(model.forward(p, X)[0]) ** 2
    mine = np.array([np.abs(Statevector(build(x, measure=False)).data) ** 2 for x in X])
    diff = np.abs(exact - mine).max()
    print(f"Test 1 - rebuilt circuit vs your trained model: biggest difference = {diff:.2e}")
    print("   PASS" if diff < 1e-8 else "   FAIL - do not go further, send this output to Claude")

    rng = np.random.default_rng(1)
    agree = 0
    for i, x in enumerate(X):
        probs = mine[i] / mine[i].sum()
        counts = {format(k, f"0{n}b"): int(c) for k, c in enumerate(rng.multinomial(SHOTS, probs)) if c}
        agree += predict_from_counts(counts) == predict_from_probs(probs)
    print(f"Test 2 - turning counts into a diagnosis: {agree}/{len(X)} match the exact answer "
          f"(small misses are normal shot noise)")


def do_run():
    if DEVICE_ID.startswith("PASTE"):
        raise SystemExit("Edit DEVICE_ID at the top of run_on_qpu.py first (use --list to find one).")
    from qbraid.runtime import QbraidProvider
    device = QbraidProvider().get_device(DEVICE_ID)
    X, y = load_patients(N_PATIENTS)
    preds = []
    for i, x in enumerate(X):
        job = device.run(build(x), shots=SHOTS)
        print(f"patient {i + 1}/{len(X)} submitted (job {getattr(job, 'id', '?')}), waiting...", flush=True)
        job.wait_for_final_state()
        counts = get_counts(job.result())
        exact = predict_from_probs(np.abs(Statevector(build(x, measure=False)).data) ** 2)
        preds.append(predict_from_counts(counts))
        print(f"   chip says: {preds[-1]} | perfect simulator says: {exact} | true: {y[i]}", flush=True)
    acc, f1 = score(y, np.array(preds))
    print(f"\n{len(y)} patients on {DEVICE_ID}: accuracy={acc:.3f}  macroF1={f1:.3f}")
    pd.DataFrame({"true": y, "predicted": preds}).to_csv("qpu_results.csv", index=False)
    print("Saved qpu_results.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.list:
        do_list()
    elif a.check:
        do_check()
    elif a.run:
        do_run()
    else:
        ap.print_help()
