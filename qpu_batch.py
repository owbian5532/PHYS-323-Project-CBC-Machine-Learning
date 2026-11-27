"""
qpu_batch.py - run the fresh-test patients on the chip WITHOUT waiting, then collect results later.

Needs run_on_qpu.py, final_train.py, Qubit_Pattern_Comparison_VQC_v2.py, cbc_train_lda.csv and the
final_model folder in the same folder. The device and shots come from DEVICE_ID / SHOTS in run_on_qpu.py.

    python qpu_batch.py --submit 10     sends the next 10 unsent patients, then returns immediately
    python qpu_batch.py --status        shows how many have been sent
    python qpu_batch.py --collect       fetches whatever has finished and prints the scores

Every sent job is written to qpu_jobs.csv right away, so a timed-out session never loses anything.
Patients are the 145-row "fresh test" set from final_train.py, in a fixed random order, so
submitting 10 now and 10 later continues down the same list.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from qiskit.quantum_info import Statevector

import final_train as ft
import run_on_qpu as r

JOBS_FILE = "qpu_jobs.csv"
OUT_FILE = "qpu_batch_results.csv"


def fresh_test_order():
    """The same fresh-test rows final_train.py held out, in a fixed random order."""
    tr = pd.read_csv(ft.TRAIN_FILE)
    feats = [c for c in tr.columns if c != ft.LABEL_COL]
    Xall, yall = tr[feats].values, tr[ft.LABEL_COL].astype(str).values
    _, fresh = ft.strat_split(np.arange(len(Xall)), yall, ft.FRESH_TEST_FRAC, ft.SPLIT_SEED)
    order = np.random.default_rng(0).permutation(fresh)
    return order, ft.scale_features(Xall[order], r.d["lo"], r.d["hi"]), yall[order]


def load_jobs():
    if os.path.exists(JOBS_FILE):
        return pd.read_csv(JOBS_FILE)
    return pd.DataFrame(columns=["row", "job_id", "device", "shots"])


def find_counts(obj):
    if isinstance(obj, dict):
        for key in ("measurementCounts", "measurement_counts", "counts"):
            if isinstance(obj.get(key), dict):
                return obj[key]
        for v in obj.values():
            found = find_counts(v)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_counts(v)
            if found:
                return found
    return None


def submit(count):
    if r.DEVICE_ID.startswith("PASTE"):
        raise SystemExit("Set DEVICE_ID in run_on_qpu.py first.")
    from qbraid.runtime import QbraidProvider
    device = QbraidProvider().get_device(r.DEVICE_ID)
    order, X, _ = fresh_test_order()
    jobs = load_jobs()
    sent = set(int(v) for v in jobs["row"])
    todo = [i for i in range(len(order)) if int(order[i]) not in sent][:count]
    print(f"Sending {len(todo)} patient(s) to {r.DEVICE_ID} at {r.SHOTS} shots each "
          f"(about {len(todo) * 102.5:.0f} credits if this is Garnet at 500 shots)", flush=True)
    for i in todo:
        job = device.run(r.build(X[i]), shots=r.SHOTS)
        line = pd.DataFrame([{"row": int(order[i]), "job_id": job.id, "device": r.DEVICE_ID, "shots": r.SHOTS}])
        line.to_csv(JOBS_FILE, mode="a", header=not os.path.exists(JOBS_FILE), index=False)
        print(f"  sent patient row {int(order[i])}: {job.id}", flush=True)
    print(f"\nDone. {len(load_jobs())} of {len(order)} patients sent in total. "
          f"Run  python qpu_batch.py --collect  later.")


def status():
    order, _, _ = fresh_test_order()
    print(f"{len(load_jobs())} of {len(order)} fresh-test patients sent so far.")


def collect():
    from qbraid_core.sessions import QbraidSession
    s = QbraidSession()
    jobs = load_jobs()
    order, X, y = fresh_test_order()
    pos = {int(o): i for i, o in enumerate(order)}
    rows, waiting = [], 0
    for _, j in jobs.iterrows():
        i = pos[int(j["row"])]
        try:
            counts = find_counts(s.get(f"/jobs/{j['job_id']}/result").json())
        except Exception:
            counts = None
        if not counts:
            waiting += 1
            continue
        P = np.zeros(2 ** r.n)
        total = sum(counts.values())
        for bits, c in counts.items():
            P[int(bits.replace(" ", ""), 2)] += c / total
        exact_p = np.abs(Statevector(r.build(X[i], measure=False)).data) ** 2
        rows.append({"row": int(j["row"]), "true": y[i], "chip": r.predict_from_probs(P),
                     "exact": r.predict_from_probs(exact_p),
                     "tvd": 0.5 * np.abs(P - exact_p).sum(), "job_id": j["job_id"]})
    print(f"{len(rows)} finished, {waiting} not finished yet (still queued, or failed).")
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, index=False)
    ca, cf = r.score(df["true"], df["chip"])
    ea, ef = r.score(df["true"], df["exact"])
    print(f"\nOn these {len(df)} patients:")
    print(f"  real chip         accuracy={ca:.3f}  macroF1={cf:.3f}")
    print(f"  perfect simulator accuracy={ea:.3f}  macroF1={ef:.3f}")
    print(f"  chip agrees with the perfect simulator on {np.mean(df['chip'] == df['exact']):.0%} of patients")
    print(f"  average distance of chip counts from the perfect answer (0 = identical, 1 = totally different): "
          f"{df['tvd'].mean():.3f}")
    print(f"Saved {OUT_FILE}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 2 and a[0] == "--submit":
        submit(int(a[1]))
    elif a == ["--status"]:
        status()
    elif a == ["--collect"]:
        collect()
    else:
        print(__doc__)
