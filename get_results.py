"""
get_results.py - fetch a FINISHED qBraid job by its ID and turn the chip's counts into a diagnosis.
Use it when the terminal session timed out before run_on_qpu.py could print the answer.

Put it in the same folder as run_on_qpu.py, then:

    python get_results.py --recent               shows your most recent jobs (to find the ID)
    python get_results.py JOB_ID                 fetches that job's result (patient 1)
    python get_results.py JOB_ID1 JOB_ID2 ...    several jobs = patients 1, 2, ... in the same order they were sent

JOB_ID looks like:  aws:iqm:qpu:garnet-xxxx-qjob-xxxxxxxx
"""
import sys
import json
import numpy as np
import pandas as pd
from qbraid_core.sessions import QbraidSession
from qiskit.quantum_info import Statevector

import run_on_qpu as r          # reuses the model, the circuit and the counts -> diagnosis step


def find_counts(obj):
    """Search any nested result for the {bitstring: count} table."""
    if isinstance(obj, dict):
        for key in ("measurementCounts", "measurement_counts", "counts"):
            if key in obj and isinstance(obj[key], dict):
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


def fetch(job_id):
    s = QbraidSession()
    resp = s.get(f"/jobs/{job_id}/result")
    return resp.json()


def show_recent():
    s = QbraidSession()
    for path in ("/jobs", "/jobs?pageSize=10"):
        try:
            print(json.dumps(s.get(path).json(), indent=1)[:6000])
            return
        except Exception as e:
            last = e
    print("Could not list jobs:", last)


def main(ids):
    X, y = r.load_patients(len(ids))
    preds = []
    for i, job_id in enumerate(ids):
        raw = fetch(job_id)
        counts = find_counts(raw)
        if not counts:
            print(f"No counts found for {job_id}. Raw reply (send this to Claude):\n{json.dumps(raw, indent=1)[:3000]}")
            continue
        exact = r.predict_from_probs(np.abs(Statevector(r.build(X[i], measure=False)).data) ** 2)
        preds.append(r.predict_from_counts(counts))
        print(f"patient {i + 1}: chip says: {preds[-1]} | perfect simulator says: {exact} | true: {y[i]}")
        print(f"   raw counts: {counts}")
    if preds:
        acc, f1 = r.score(y[:len(preds)], np.array(preds))
        print(f"\n{len(preds)} patient(s): accuracy={acc:.3f}  macroF1={f1:.3f}")
        pd.DataFrame({"true": y[:len(preds)], "predicted": preds}).to_csv("qpu_results.csv", index=False)
        print("Saved qpu_results.csv")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
    elif args[0] == "--recent":
        show_recent()
    else:
        main(args)
