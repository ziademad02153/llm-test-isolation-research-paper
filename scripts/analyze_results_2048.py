"""
Comprehensive Analysis Script for results_2048/
Computes:
1. Descriptive Statistics (Usability, Detection, Conditional Detection, Tokens, Finish Reasons).
2. Side-by-side comparison with results/ (512-token pilot).
3. Task-level Paired t-tests and Wilcoxon Signed-Rank tests (N=15 clusters, df=14).
4. Liang-Zeger Clustered Logistic Regression with Interaction Term (df=14).
5. Formal Mediation Analysis (Order 1, Order 2, Symmetric Mean) + 2,000 Cluster Bootstrap Resamples.
6. Per-task breakdown table across all 15 problem instances.
"""

import json
import csv
import math
import random
from collections import defaultdict, Counter
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
RES_2048_DIR = ROOT / "results_2048"
RES_512_DIR = ROOT / "results"

# Load 2048 records
records_2048_path = RES_2048_DIR / "experiment_records.jsonl"
with open(records_2048_path, "r", encoding="utf-8") as f:
    records_2048 = [json.loads(line) for line in f]

# Load 512 records
records_512_path = RES_512_DIR / "experiment_records.csv"
with open(records_512_path, "r", encoding="utf-8") as f:
    records_512 = list(csv.DictReader(f))

CONDITIONS = ["T0_Spec_Same", "T1_Code_Same", "T2_Code_Cross", "T3_Spec_Cross"]
TASKS = sorted(list(set(r["problem_id"] for r in records_2048)))
assert len(TASKS) == 15, f"Expected 15 tasks, got {len(TASKS)}"

print("================================================================================")
print("                       ANALYSIS OF RESULTS_2048 DATASET                        ")
print("================================================================================")

# 1. Descriptive stats for 2048
desc_2048 = {}
for c in CONDITIONS:
    sub = [r for r in records_2048 if r["condition"] == c]
    n = len(sub)
    n_usable = sum(1 for r in sub if r["is_usable"] is True)
    n_detect = sum(1 for r in sub if r["detects_defect"] is True)
    comp_tokens = [r.get("completion_tokens", 0) for r in sub]
    prompt_tokens = [r.get("prompt_tokens", 0) for r in sub]
    fr = Counter(r.get("finish_reason") for r in sub)
    n_syntax = sum(1 for r in sub if r.get("has_syntax_error") is True)
    n_empty = sum(1 for r in sub if r.get("is_empty_suite") is True)
    n_zero_assert = sum(1 for r in sub if r.get("total_assertions", 0) == 0)
    
    desc_2048[c] = {
        "N": n,
        "n_usable": n_usable,
        "usable_pct": n_usable / n * 100,
        "n_detect": n_detect,
        "detect_pct": n_detect / n * 100,
        "cond_detect_pct": (n_detect / n_usable * 100) if n_usable else 0.0,
        "avg_comp_tokens": sum(comp_tokens) / n,
        "min_comp_tokens": min(comp_tokens),
        "max_comp_tokens": max(comp_tokens),
        "avg_prompt_tokens": sum(prompt_tokens) / n,
        "finish_reasons": dict(fr),
        "syntax_errors": n_syntax,
        "empty_suites": n_empty,
        "zero_assertions": n_zero_assert
    }

# 2. Side-by-Side Comparison: 512 vs 2048
print("\n--- 1. DESCRIPTIVE COMPARISON: 512 (PILOT) vs 2048 (PRIMARY) ---")
print(f"{'Condition':<15} | {'Metric':<18} | {'512-Token':<12} | {'2048-Token':<12} | {'Delta':<10}")
print("-" * 75)

desc_512 = {}
for c in CONDITIONS:
    sub = [r for r in records_512 if r["condition"] == c]
    n = len(sub)
    n_usable = sum(1 for r in sub if r["is_usable"] == "True")
    n_detect = sum(1 for r in sub if r["detects_defect"] == "True")
    desc_512[c] = {
        "usable_pct": n_usable / n * 100,
        "detect_pct": n_detect / n * 100,
        "cond_detect_pct": (n_detect / n_usable * 100) if n_usable else 0.0
    }

for c in CONDITIONS:
    u512 = desc_512[c]["usable_pct"]
    u2048 = desc_2048[c]["usable_pct"]
    d512 = desc_512[c]["detect_pct"]
    d2048 = desc_2048[c]["detect_pct"]
    cd512 = desc_512[c]["cond_detect_pct"]
    cd2048 = desc_2048[c]["cond_detect_pct"]
    
    print(f"{c:<15} | Usability (%)      | {u512:6.2f}%      | {u2048:6.2f}%      | {u2048-u512:+6.2f} pp")
    print(f"{'':<15} | Defect Detect (%)  | {d512:6.2f}%      | {d2048:6.2f}%      | {d2048-d512:+6.2f} pp")
    print(f"{'':<15} | Cond Detect (%)    | {cd512:6.2f}%      | {cd2048:6.2f}%      | {cd2048-cd512:+6.2f} pp")
    print("-" * 75)

# 3. Task-level Paired Analysis (N=15 clusters)
print("\n--- 2. TASK-LEVEL PAIRED INFERENCE (N=15 clusters, df=14) ---")
# Per task detection rates
task_detect = defaultdict(dict)
for t in TASKS:
    for c in CONDITIONS:
        rows = [r for r in records_2048 if r["problem_id"] == t and r["condition"] == c]
        d_count = sum(1 for r in rows if r["detects_defect"] is True)
        task_detect[t][c] = d_count / len(rows)

def paired_t_test(c1, c2):
    diffs = [task_detect[t][c2] - task_detect[t][c1] for t in TASKS]
    mean_diff = sum(diffs) / len(diffs)
    variance = sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)
    se = math.sqrt(variance / len(diffs))
    t_stat = mean_diff / se if se > 0 else 0.0
    
    # Exact Gamma(7.5)/(sqrt(14*pi)*Gamma(7)) = 0.3918874558
    c_const = 0.3918874558
    limit = abs(t_stat)
    steps = 100000
    b = 50.0
    if limit < b:
        dx = (b - limit) / steps
        area = sum((1.0 + (limit + (i + 0.5) * dx)**2 / 14)**(-7.5) * dx for i in range(steps))
        p_val = 2.0 * c_const * area
    else:
        p_val = 0.0
    
    # Wilcoxon signed-rank
    non_zero = [(abs(d), 1 if d > 0 else -1) for d in diffs if abs(d) > 1e-9]
    non_zero.sort(key=lambda x: x[0])
    # assign ranks
    ranks = []
    i = 0
    while i < len(non_zero):
        j = i
        while j < len(non_zero) and abs(non_zero[j][0] - non_zero[i][0]) < 1e-9:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks.append((avg_rank, non_zero[k][1]))
        i = j
    w_pos = sum(r * (1 if s > 0 else 0) for r, s in ranks)
    w_neg = sum(r * (1 if s < 0 else 0) for r, s in ranks)
    w_stat = min(w_pos, w_neg)
    
    # 95% CI: mean_diff +- 2.1448 * se
    ci_low = mean_diff - 2.1448 * se
    ci_high = mean_diff + 2.1448 * se
    return {
        "mean_diff_pp": mean_diff * 100,
        "se_pp": se * 100,
        "ci_95_pp": (ci_low * 100, ci_high * 100),
        "t_stat": t_stat,
        "p_val": p_val,
        "w_stat": w_stat,
        "n_pairs": len(diffs)
    }

pairs_to_test = [
    ("T0_vs_T1", "T0_Spec_Same", "T1_Code_Same", "RQ1: Code Exposure (GPT)"),
    ("T1_vs_T2", "T1_Code_Same", "T2_Code_Cross", "RQ2: Model Diversity (under Code Exposure)"),
    ("T0_vs_T3", "T0_Spec_Same", "T3_Spec_Cross", "RQ3a: Model Diversity (under Spec Only)"),
    ("T3_vs_T2", "T3_Spec_Cross", "T2_Code_Cross", "RQ3b: Code Exposure (Qwen)")
]

for pkey, c1, c2, title in pairs_to_test:
    res = paired_t_test(c1, c2)
    print(f"\n{title} ({c1} -> {c2}):")
    print(f"  Mean Delta: {res['mean_diff_pp']:+.2f} pp (SE = {res['se_pp']:.2f} pp)")
    print(f"  95% CI: [{res['ci_95_pp'][0]:+.2f} pp, {res['ci_95_pp'][1]:+.2f} pp]")
    print(f"  Paired t(14) = {res['t_stat']:.4f}, p = {res['p_val']:.6f} ({'p < 0.001' if res['p_val'] < 0.001 else f'p = {res['p_val']:.4f}'})")
    print(f"  Wilcoxon W = {res['w_stat']}")

# 4. Liang-Zeger Clustered Logistic Regression
print("\n--- 3. LIANG-ZEGER CLUSTERED LOGISTIC REGRESSION (df=14) ---")
# Model: logit(Y) = beta0 + beta1*Exp + beta2*Div + beta3*(Exp*Div)
# Coding: Exp = 1 if Code, 0 if Spec; Div = 1 if Cross (Qwen), 0 if Same (GPT)
def get_x(r):
    exp = 1.0 if not r["is_spec_only"] is True else 0.0
    div = 1.0 if r["is_cross_family"] is True else 0.0
    return [1.0, exp, div, exp * div]

X = [get_x(r) for r in records_2048]
Y = [1.0 if r["detects_defect"] is True else 0.0 for r in records_2048]
N = len(records_2048)
K = 4
G = len(TASKS)

# Fit by Newton-Raphson
beta = [0.0, 0.0, 0.0, 0.0]
for iteration in range(25):
    P = [1.0 / (1.0 + math.exp(-sum(xi * bi for xi, bi in zip(x, beta)))) for x in X]
    W = [p * (1.0 - p) for p in P]
    grad = [0.0] * 4
    for i in range(N):
        e_i = Y[i] - P[i]
        for j in range(4):
            grad[j] += X[i][j] * e_i
    H = [[0.0] * 4 for _ in range(4)]
    for i in range(N):
        w_i = W[i]
        for j in range(4):
            for k in range(4):
                H[j][k] += w_i * X[i][j] * X[i][k]
    # Invert H
    M = [H[i][:] + [1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    for i in range(4):
        pivot = M[i][i]
        for j in range(8):
            M[i][j] /= pivot
        for r in range(4):
            if r != i:
                factor = M[r][i]
                for j in range(8):
                    M[r][j] -= factor * M[i][j]
    H_inv = [[M[i][j + 4] for j in range(4)] for i in range(4)]
    step = [sum(H_inv[j][k] * grad[k] for k in range(4)) for j in range(4)]
    for j in range(4):
        beta[j] += step[j]
    if max(abs(s) for s in step) < 1e-8:
        break

# Clustered Sandwich Variance
P = [1.0 / (1.0 + math.exp(-sum(xi * bi for xi, bi in zip(x, beta)))) for x in X]
W = [p * (1.0 - p) for p in P]
B = [[0.0] * 4 for _ in range(4)]
dfc = (G / (G - 1.0)) * ((N - 1.0) / (N - K))
for g in TASKS:
    s_g = [0.0] * 4
    for i, r in enumerate(records_2048):
        if r["problem_id"] == g:
            e_i = Y[i] - P[i]
            for j in range(4):
                s_g[j] += X[i][j] * e_i
    for j in range(4):
        for k in range(4):
            B[j][k] += s_g[j] * s_g[k]
for j in range(4):
    for k in range(4):
        B[j][k] *= dfc

Temp = [[sum(B[i][k] * H_inv[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
V_cluster = [[sum(H_inv[i][k] * Temp[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
se_cluster = [math.sqrt(V_cluster[i][i]) for i in range(4)]
t_cluster = [beta[i] / se_cluster[i] for i in range(4)]

def calc_p_t14(t_val):
    c_const = 0.3918874558
    limit = abs(t_val)
    steps = 100000
    b = 50.0
    if limit < b:
        dx = (b - limit) / steps
        area = sum((1.0 + (limit + (i + 0.5) * dx)**2 / 14)**(-7.5) * dx for i in range(steps))
        return 2.0 * c_const * area
    return 0.0

p_cluster = [calc_p_t14(t) for t in t_cluster]
param_names = ["Intercept (beta_0)", "Code Exposure (beta_1)", "Model Diversity (beta_2)", "Interaction (beta_3)"]

print(f"{'Parameter':<25} | {'Estimate':<10} | {'Cluster SE':<12} | {'t (df=14)':<10} | {'p-value':<10} | {'95% CI'}")
print("-" * 85)
for i in range(4):
    ci_low = beta[i] - 2.1448 * se_cluster[i]
    ci_high = beta[i] + 2.1448 * se_cluster[i]
    p_str = f"< 0.0001" if p_cluster[i] < 0.0001 else f"{p_cluster[i]:.4f}"
    print(f"{param_names[i]:<25} | {beta[i]:9.5f} | {se_cluster[i]:11.5f} | {t_cluster[i]:9.4f} | {p_str:<10} | [{ci_low:+.4f}, {ci_high:+.4f}]")

# 5. Causal Mediation Analysis
print("\n--- 4. CAUSAL MEDIATION ANALYSIS (Dual Potential-Outcomes Decompositions) ---")
# Group records by task and condition
by_task = defaultdict(lambda: defaultdict(list))
for r in records_2048:
    by_task[r["problem_id"]][r["condition"]].append(r)

def compute_decomp(task_list, c_base, c_focal):
    b_rows = [r for t in task_list for r in by_task[t][c_base]]
    f_rows = [r for t in task_list for r in by_task[t][c_focal]]
    n0, n1 = len(b_rows), len(f_rows)
    m0_count = sum(1 for r in b_rows if r["is_usable"] is True)
    m1_count = sum(1 for r in f_rows if r["is_usable"] is True)
    y0_count = sum(1 for r in b_rows if r["detects_defect"] is True)
    y1_count = sum(1 for r in f_rows if r["detects_defect"] is True)
    
    m0 = m0_count / n0 if n0 else 0.0
    m1 = m1_count / n1 if n1 else 0.0
    q0 = y0_count / m0_count if m0_count else 0.0
    q1 = y1_count / m1_count if m1_count else 0.0
    
    e00 = q0 * m0
    e11 = q1 * m1
    te = e11 - e00
    
    # Order 1: M(0) baseline
    nde_0 = (q1 - q0) * m0
    acme_1 = q1 * (m1 - m0)
    pm_1 = (acme_1 / te) * 100 if abs(te) > 1e-6 else 0.0
    
    # Order 2: M(1) baseline
    acme_0 = q0 * (m1 - m0)
    nde_1 = (q1 - q0) * m1
    pm_0 = (acme_0 / te) * 100 if abs(te) > 1e-6 else 0.0
    
    # Symmetric Mean
    nde_mean = 0.5 * (nde_0 + nde_1)
    acme_mean = 0.5 * (acme_1 + acme_0)
    pm_mean = (acme_mean / te) * 100 if abs(te) > 1e-6 else 0.0
    
    return {
        "m0": m0, "m1": m1, "q0": q0, "q1": q1,
        "TE": te,
        "Order1_NDE": nde_0, "Order1_ACME": acme_1, "Order1_PM": pm_1,
        "Order2_NDE": nde_1, "Order2_ACME": acme_0, "Order2_PM": pm_0,
        "Mean_NDE": nde_mean, "Mean_ACME": acme_mean, "Mean_PM": pm_mean
    }

gpt_med = compute_decomp(TASKS, "T0_Spec_Same", "T1_Code_Same")
qwen_med = compute_decomp(TASKS, "T3_Spec_Cross", "T2_Code_Cross")

print("GPT Pathway (T0 -> T1):")
print(f"  m0 = {gpt_med['m0']:.4f} ({gpt_med['m0']*100:.2f}%), m1 = {gpt_med['m1']:.4f} ({gpt_med['m1']*100:.2f}%)")
print(f"  q0 = {gpt_med['q0']:.4f} ({gpt_med['q0']*100:.2f}%), q1 = {gpt_med['q1']:.4f} ({gpt_med['q1']*100:.2f}%)")
print(f"  Total Effect (TE) = {gpt_med['TE']:+.4f} ({gpt_med['TE']*100:+.2f} pp)")
print(f"  Order 1 (M(0) baseline): ACME(1) = {gpt_med['Order1_ACME']:+.4f}, NDE(0) = {gpt_med['Order1_NDE']:+.4f}, PM = {gpt_med['Order1_PM']:.2f}%")
print(f"  Order 2 (M(1) baseline): ACME(0) = {gpt_med['Order2_ACME']:+.4f}, NDE(1) = {gpt_med['Order2_NDE']:+.4f}, PM = {gpt_med['Order2_PM']:.2f}%")
print(f"  Symmetric Mean:          ACME    = {gpt_med['Mean_ACME']:+.4f}, NDE    = {gpt_med['Mean_NDE']:+.4f}, PM = {gpt_med['Mean_PM']:.2f}%")

print("\nQwen Pathway (T3 -> T2):")
print(f"  m0 = {qwen_med['m0']:.4f} ({qwen_med['m0']*100:.2f}%), m1 = {qwen_med['m1']:.4f} ({qwen_med['m1']*100:.2f}%)")
print(f"  q0 = {qwen_med['q0']:.4f} ({qwen_med['q0']*100:.2f}%), q1 = {qwen_med['q1']:.4f} ({qwen_med['q1']*100:.2f}%)")
print(f"  Total Effect (TE) = {qwen_med['TE']:+.4f} ({qwen_med['TE']*100:+.2f} pp)")
print(f"  Order 1 (M(0) baseline): ACME(1) = {qwen_med['Order1_ACME']:+.4f}, NDE(0) = {qwen_med['Order1_NDE']:+.4f}, PM = {qwen_med['Order1_PM']:.2f}%")
print(f"  Order 2 (M(1) baseline): ACME(0) = {qwen_med['Order2_ACME']:+.4f}, NDE(1) = {qwen_med['Order2_NDE']:+.4f}, PM = {qwen_med['Order2_PM']:.2f}%")
print(f"  Symmetric Mean:          ACME    = {qwen_med['Mean_ACME']:+.4f}, NDE    = {qwen_med['Mean_NDE']:+.4f}, PM = {qwen_med['Mean_PM']:.2f}%")

# Cluster Bootstrap Resampling (B=2,000)
print("\nRunning Cluster Bootstrap (B=2,000 iterations, seed=42)...")
random.seed(42)
B = 2000
boot_gpt_pm1 = []
boot_gpt_pm0 = []
boot_gpt_pm_mean = []
boot_gpt_te = []

boot_qwen_pm1 = []
boot_qwen_pm0 = []
boot_qwen_pm_mean = []
boot_qwen_te = []

for b in range(B):
    resampled_tasks = [random.choice(TASKS) for _ in range(len(TASKS))]
    d_g = compute_decomp(resampled_tasks, "T0_Spec_Same", "T1_Code_Same")
    d_q = compute_decomp(resampled_tasks, "T3_Spec_Cross", "T2_Code_Cross")
    
    boot_gpt_te.append(d_g["TE"])
    boot_gpt_pm1.append(d_g["Order1_PM"])
    boot_gpt_pm0.append(d_g["Order2_PM"])
    boot_gpt_pm_mean.append(d_g["Mean_PM"])
    
    boot_qwen_te.append(d_q["TE"])
    boot_qwen_pm1.append(d_q["Order1_PM"])
    boot_qwen_pm0.append(d_q["Order2_PM"])
    boot_qwen_pm_mean.append(d_q["Mean_PM"])

def get_percentile_ci(arr, low=2.5, high=97.5):
    s = sorted(arr)
    n = len(s)
    idx_l = int(math.floor(n * (low / 100.0)))
    idx_h = int(math.ceil(n * (high / 100.0))) - 1
    return s[idx_l], s[idx_h]

gpt_pm_mean_ci = get_percentile_ci(boot_gpt_pm_mean)
gpt_pm1_ci = get_percentile_ci(boot_gpt_pm1)
gpt_pm0_ci = get_percentile_ci(boot_gpt_pm0)

print(f"Bootstrap 95% CIs for GPT (T0 -> T1):")
print(f"  Order 1 PM: {gpt_med['Order1_PM']:.2f}%  [95% CI: {gpt_pm1_ci[0]:.2f}%, {gpt_pm1_ci[1]:.2f}%]")
print(f"  Order 2 PM: {gpt_med['Order2_PM']:.2f}%  [95% CI: {gpt_pm0_ci[0]:.2f}%, {gpt_pm0_ci[1]:.2f}%]")
print(f"  Mean PM:    {gpt_med['Mean_PM']:.2f}%  [95% CI: {gpt_pm_mean_ci[0]:.2f}%, {gpt_pm_mean_ci[1]:.2f}%]")

# 6. Per-Task Breakdown Table
print("\n--- 5. PER-TASK BREAKDOWN (Defect Detections / 15 trials per condition) ---")
print(f"{'Task ID':<12} | {'Category':<15} | {'Difficulty':<10} | {'T0 (Spec)':<10} | {'T1 (Code)':<10} | {'T2 (Q-Cd)':<10} | {'T3 (Q-Sp)':<10}")
print("-" * 85)

tasks_meta = {}
for r in records_2048:
    tid = r["problem_id"]
    if tid not in tasks_meta:
        # Load from benchmark tasks if available or infer
        tasks_meta[tid] = {"id": tid}

# Read benchmark metadata if exists
meta_path = ROOT / "data" / "verified_tasks_manifest.json"
if meta_path.exists():
    with open(meta_path, "r", encoding="utf-8") as f:
        bench_tasks = json.load(f)
        for bt in bench_tasks:
            if bt["problem_id"] in tasks_meta:
                tasks_meta[bt["problem_id"]]["category"] = bt.get("category", "General")
                tasks_meta[bt["problem_id"]]["difficulty"] = bt.get("difficulty", "Medium")
                tasks_meta[bt["problem_id"]]["fault_type"] = bt.get("fault_type", "General")

task_table_data = []
for t in TASKS:
    cat = tasks_meta.get(t, {}).get("category", "Unknown")
    diff = tasks_meta.get(t, {}).get("difficulty", "Unknown")
    t0_d = sum(1 for r in records_2048 if r["problem_id"] == t and r["condition"] == "T0_Spec_Same" and r["detects_defect"] is True)
    t1_d = sum(1 for r in records_2048 if r["problem_id"] == t and r["condition"] == "T1_Code_Same" and r["detects_defect"] is True)
    t2_d = sum(1 for r in records_2048 if r["problem_id"] == t and r["condition"] == "T2_Code_Cross" and r["detects_defect"] is True)
    t3_d = sum(1 for r in records_2048 if r["problem_id"] == t and r["condition"] == "T3_Spec_Cross" and r["detects_defect"] is True)
    
    print(f"{t:<12} | {cat:<15} | {diff:<10} | {t0_d:2d}/15       | {t1_d:2d}/15       | {t2_d:2d}/15       | {t3_d:2d}/15")
    task_table_data.append({
        "task_id": t, "category": cat, "difficulty": diff,
        "t0": t0_d, "t1": t1_d, "t2": t2_d, "t3": t3_d
    })

# Delegate artifact generation to generate_dual_analysis_summary to maintain dual-harness schema
import sys
sys.path.insert(0, str(ROOT))
from scripts.generate_dual_analysis_summary import generate_dual_summary

generate_dual_summary()
