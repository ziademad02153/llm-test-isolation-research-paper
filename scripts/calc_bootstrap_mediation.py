"""
Reproducible Cluster Bootstrap Mediation Analysis for LLM Test Generation Experiment.
Implements dual potential-outcomes mediation decompositions (Order 1 & Order 2)
with task-cluster resampling (B=2,000 iterations) using Python standard library.
Outputs exact point estimates, 95% percentile bootstrap CIs, and saves JSON artifact.
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

# 1. Load 900 raw experimental evaluation records
records_path = Path("results/experiment_records.csv")
with open(records_path, "r", encoding="utf-8") as f:
    records = list(csv.DictReader(f))

# Group records by task (15 clusters) and experimental condition
by_task = defaultdict(lambda: defaultdict(list))
for r in records:
    by_task[r["problem_id"]][r["condition"]].append(r)

tasks = sorted(list(by_task.keys()))
assert len(tasks) == 15, f"Expected 15 task clusters, found {len(tasks)}"

def compute_decomposition(task_list, c_base, c_focal):
    """
    Computes potential-outcomes counterfactual expectations and mediation effects:
    Let q_t = P(Y=1 | T=t, M=1) and m_t = P(M=1 | T=t).
    TE = q_1 * m_1 - q_0 * m_0
    Order 1 (M(0) baseline): NDE(0) = (q_1 - q_0) * m_0, ACME(1) = q_1 * (m_1 - m_0)
    Order 2 (M(1) baseline): ACME(0) = q_0 * (m_1 - m_0), NDE(1) = (q_1 - q_0) * m_1
    """
    base_rows = []
    focal_rows = []
    for t in task_list:
        base_rows.extend(by_task[t][c_base])
        focal_rows.extend(by_task[t][c_focal])

    n_0 = len(base_rows)
    n_1 = len(focal_rows)

    m0_count = sum(1 for r in base_rows if r["is_usable"] == "True")
    m1_count = sum(1 for r in focal_rows if r["is_usable"] == "True")

    y0_count = sum(1 for r in base_rows if r["detects_defect"] == "True")
    y1_count = sum(1 for r in focal_rows if r["detects_defect"] == "True")

    # Usability probabilities m_t = P(M=1 | T=t)
    m_0 = m0_count / n_0 if n_0 > 0 else 0.0
    m_1 = m1_count / n_1 if n_1 > 0 else 0.0

    # Conditional detection probabilities q_t = P(Y=1 | T=t, M=1)
    q_0 = y0_count / m0_count if m0_count > 0 else 0.0
    q_1 = y1_count / m1_count if m1_count > 0 else 0.0

    # Unconditional detection e_tt = P(Y=1 | T=t) = q_t * m_t (since Y <= M)
    e_00 = q_0 * m_0
    e_11 = q_1 * m_1
    te = e_11 - e_00

    # Order 1 (Specification baseline M(0))
    nde_0 = (q_1 - q_0) * m_0
    acme_1 = q_1 * (m_1 - m_0)
    pm_1 = (acme_1 / te) * 100.0 if abs(te) > 1e-6 else 0.0

    # Order 2 (Code-exposure baseline M(1))
    acme_0 = q_0 * (m_1 - m_0)
    nde_1 = (q_1 - q_0) * m_1
    pm_0 = (acme_0 / te) * 100.0 if abs(te) > 1e-6 else 0.0

    # Symmetric Mean Decomposition
    nde_mean = 0.5 * (nde_0 + nde_1)
    acme_mean = 0.5 * (acme_1 + acme_0)
    pm_mean = (acme_mean / te) * 100.0 if abs(te) > 1e-6 else 0.0

    return {
        "TE": te,
        "Order1_NDE": nde_0,
        "Order1_ACME": acme_1,
        "Order1_PM": pm_1,
        "Order2_NDE": nde_1,
        "Order2_ACME": acme_0,
        "Order2_PM": pm_0,
        "Mean_NDE": nde_mean,
        "Mean_ACME": acme_mean,
        "Mean_PM": pm_mean,
        "q_0": q_0,
        "q_1": q_1,
        "m_0": m_0,
        "m_1": m_1,
    }

# 2. Main point estimates on the complete dataset
pt_gpt = compute_decomposition(tasks, "T0_Spec_Same", "T1_Code_Same")
pt_qwen = compute_decomposition(tasks, "T3_Spec_Cross", "T2_Code_Cross")

# 3. Cluster Bootstrap Resampling
RANDOM_SEED = 42
B = 2000
random.seed(RANDOM_SEED)

boot_gpt = []
boot_qwen = []

for _ in range(B):
    sampled_tasks = [random.choice(tasks) for _ in range(len(tasks))]
    boot_gpt.append(compute_decomposition(sampled_tasks, "T0_Spec_Same", "T1_Code_Same"))
    boot_qwen.append(compute_decomposition(sampled_tasks, "T3_Spec_Cross", "T2_Code_Cross"))

def get_percentile_ci(replicates, key):
    values = sorted([r[key] for r in replicates])
    low_idx = int(0.025 * len(values))
    high_idx = int(0.975 * len(values))
    return values[low_idx], values[high_idx]

# Construct comprehensive audit dictionary
results = {
    "metadata": {
        "random_seed": RANDOM_SEED,
        "replicates_B": B,
        "cluster_count": len(tasks),
        "total_evaluations": len(records),
        "valid_replicates_gpt": len(boot_gpt),
        "valid_replicates_qwen": len(boot_qwen),
    },
    "gpt_oss_120b": {},
    "qwen3.8_27b": {}
}

metrics = [
    "TE", "Order1_NDE", "Order1_ACME", "Order1_PM",
    "Order2_NDE", "Order2_ACME", "Order2_PM",
    "Mean_NDE", "Mean_ACME", "Mean_PM"
]

for m in metrics:
    low, high = get_percentile_ci(boot_gpt, m)
    results["gpt_oss_120b"][m] = {
        "point": pt_gpt[m],
        "ci_95_low": low,
        "ci_95_high": high
    }

for m in metrics:
    low, high = get_percentile_ci(boot_qwen, m)
    results["qwen3.8_27b"][m] = {
        "point": pt_qwen[m],
        "ci_95_low": low,
        "ci_95_high": high
    }

# Save machine-readable JSON artifact
json_out_path = Path("results/bootstrap_mediation.json")
with open(json_out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"=== CLUSTER BOOTSTRAP MEDIATION REPORT (Seed={RANDOM_SEED}, B={B}) ===")
print(f"Artifact written to: {json_out_path}\n")

print("--- PRIMARY MODEL: gpt-oss-120b (T0 -> T1) ---")
g = results["gpt_oss_120b"]
print(f"Total Effect (TE):       {g['TE']['point']:+.4f} [{g['TE']['ci_95_low']:+.4f}, {g['TE']['ci_95_high']:+.4f}]")
print(f"Order 1 NDE(0):          {g['Order1_NDE']['point']:+.4f} [{g['Order1_NDE']['ci_95_low']:+.4f}, {g['Order1_NDE']['ci_95_high']:+.4f}]")
print(f"Order 1 ACME(1):         {g['Order1_ACME']['point']:+.4f} [{g['Order1_ACME']['ci_95_low']:+.4f}, {g['Order1_ACME']['ci_95_high']:+.4f}]")
print(f"Order 1 PM:              {g['Order1_PM']['point']:.2f}% [{g['Order1_PM']['ci_95_low']:.1f}%, {g['Order1_PM']['ci_95_high']:.1f}%]")
print(f"Order 2 NDE(1):          {g['Order2_NDE']['point']:+.4f} [{g['Order2_NDE']['ci_95_low']:+.4f}, {g['Order2_NDE']['ci_95_high']:+.4f}]")
print(f"Order 2 ACME(0):         {g['Order2_ACME']['point']:+.4f} [{g['Order2_ACME']['ci_95_low']:+.4f}, {g['Order2_ACME']['ci_95_high']:+.4f}]")
print(f"Order 2 PM:              {g['Order2_PM']['point']:.2f}% [{g['Order2_PM']['ci_95_low']:.1f}%, {g['Order2_PM']['ci_95_high']:.1f}%]")
print(f"Mean NDE:                {g['Mean_NDE']['point']:+.4f} [{g['Mean_NDE']['ci_95_low']:+.4f}, {g['Mean_NDE']['ci_95_high']:+.4f}]")
print(f"Mean ACME:               {g['Mean_ACME']['point']:+.4f} [{g['Mean_ACME']['ci_95_low']:+.4f}, {g['Mean_ACME']['ci_95_high']:+.4f}]")
print(f"Mean PM:                 {g['Mean_PM']['point']:.2f}% [{g['Mean_PM']['ci_95_low']:.1f}%, {g['Mean_PM']['ci_95_high']:.1f}%]")

print("\n--- SECONDARY MODEL: qwen3.8-27b (T3 -> T2) ---")
q = results["qwen3.8_27b"]
print(f"Total Effect (TE):       {q['TE']['point']:+.4f} [{q['TE']['ci_95_low']:+.4f}, {q['TE']['ci_95_high']:+.4f}]")
print(f"Order 1 NDE(0):          {q['Order1_NDE']['point']:+.4f} [{q['Order1_NDE']['ci_95_low']:+.4f}, {q['Order1_NDE']['ci_95_high']:+.4f}]")
print(f"Order 1 ACME(1):         {q['Order1_ACME']['point']:+.4f} [{q['Order1_ACME']['ci_95_low']:+.4f}, {q['Order1_ACME']['ci_95_high']:+.4f}]")
print(f"Order 2 NDE(1):          {q['Order2_NDE']['point']:+.4f} [{q['Order2_NDE']['ci_95_low']:+.4f}, {q['Order2_NDE']['ci_95_high']:+.4f}]")
print(f"Order 2 ACME(0):         {q['Order2_ACME']['point']:+.4f} [{q['Order2_ACME']['ci_95_low']:+.4f}, {q['Order2_ACME']['ci_95_high']:+.4f}]")
