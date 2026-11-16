"""
Comprehensive Causal & Statistical Analysis Engine for 900 Evaluation Records.
Computes:
1. Two-level descriptive statistics (Usability, Defect Detection, Conditional Detection).
2. Paired Exact McNemar tests across matched trial quadruplets (N=225).
3. Two One-Sided Tests (TOST) for statistical equivalence.
4. Newton-Raphson Logistic Regression with Interaction Term (Code Exposure x Model Diversity).
5. Formal Causal Mediation Analysis (ACME, NDE, Total Effect, Proportion Mediated).
6. Breakdown by problem category and difficulty.
"""

import os
import sys
import json
import csv
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.stats_utils import (
    compute_contingency_table,
    compute_tost_paired_proportions
)


def load_dataset() -> List[Dict[str, Any]]:
    csv_path = project_root / "results" / "experiment_records.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def run_descriptive_analysis(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats = {}
    for c in ["T0_Spec_Same", "T1_Code_Same", "T2_Code_Cross", "T3_Spec_Cross"]:
        cond_rows = [r for r in records if r["condition"] == c]
        n_total = len(cond_rows)
        n_usable = sum(1 for r in cond_rows if r["is_usable"] == "True")
        n_detect = sum(1 for r in cond_rows if r["detects_defect"] == "True")
        
        usable_rate = (n_usable / n_total) * 100 if n_total else 0.0
        detect_rate = (n_detect / n_total) * 100 if n_total else 0.0
        cond_detect_rate = (n_detect / n_usable) * 100 if n_usable else 0.0
        
        stats[c] = {
            "n_total": n_total,
            "n_usable": n_usable,
            "usable_pct": usable_rate,
            "n_detect": n_detect,
            "detect_pct": detect_rate,
            "cond_detect_pct": cond_detect_rate
        }
    return stats


def run_paired_comparisons(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Group by matched quadruplet key: (problem_id, repetition_number)
    quads = defaultdict(dict)
    for r in records:
        tid = r["trial_id"]
        # trial_id: TASK_XXX_Cond_RYY
        parts = tid.split("_")
        pid = parts[0] + "_" + parts[1]
        rep = parts[-1]
        quad_key = f"{pid}_{rep}"
        quads[quad_key][r["condition"]] = r

    n_quads = len(quads)
    
    pairs_def = [
        ("T0_vs_T1", "T0_Spec_Same", "T1_Code_Same", "Code Exposure under Same Model"),
        ("T1_vs_T2", "T1_Code_Same", "T2_Code_Cross", "Model Diversity under Code Exposure"),
        ("T0_vs_T3", "T0_Spec_Same", "T3_Spec_Cross", "Model Diversity under Spec Only"),
        ("T3_vs_T2", "T3_Spec_Cross", "T2_Code_Cross", "Code Exposure under Cross Model")
    ]
    
    comparisons = {}
    for comp_key, c_ref, c_focal, desc in pairs_def:
        ref_binary = []
        focal_binary = []
        for qk, cond_dict in quads.items():
            if c_ref in cond_dict and c_focal in cond_dict:
                ref_binary.append(1 if cond_dict[c_ref]["detects_defect"] == "True" else 0)
                focal_binary.append(1 if cond_dict[c_focal]["detects_defect"] == "True" else 0)
                
        # 1. McNemar Exact Test & Odds Ratio
        mcnemar_res = compute_contingency_table(ref_binary, focal_binary)
        
        # 2. TOST Equivalence Test (Margin = 0.08)
        tost_res = compute_tost_paired_proportions(ref_binary, focal_binary, margin=0.08)
        
        comparisons[comp_key] = {
            "description": desc,
            "c_ref": c_ref,
            "c_focal": c_focal,
            "n_pairs": len(ref_binary),
            "ref_successes": sum(ref_binary),
            "focal_successes": sum(focal_binary),
            "ref_rate": (sum(ref_binary) / len(ref_binary)) * 100,
            "focal_rate": (sum(focal_binary) / len(focal_binary)) * 100,
            "delta_pp": ((sum(focal_binary) - sum(ref_binary)) / len(ref_binary)) * 100,
            "mcnemar": mcnemar_res,
            "tost": tost_res
        }
    return comparisons


def fit_logistic_model(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fits: logit(P(Detect)) = b0 + b_exp * Exposure + b_div * Diversity + b_inter * (Exposure * Diversity)
    Using Newton-Raphson IRLS.
    """
    X = []
    y = []
    for r in records:
        exp = 1.0 if r["condition"] in ("T1_Code_Same", "T2_Code_Cross") else 0.0
        div = 1.0 if r["condition"] in ("T2_Code_Cross", "T3_Spec_Cross") else 0.0
        inter = exp * div
        X.append([1.0, exp, div, inter])
        y.append(1.0 if r["detects_defect"] == "True" else 0.0)

    n = len(y)
    p = 4
    beta = [0.0] * p

    for it in range(30):
        # Probabilities
        pi = []
        for i in range(n):
            eta = sum(X[i][j] * beta[j] for j in range(p))
            # numerically safe sigmoid
            if eta > 20:
                prob = 1.0
            elif eta < -20:
                prob = 0.0
            else:
                prob = 1.0 / (1.0 + math.exp(-eta))
            pi.append(max(1e-12, min(1.0 - 1e-12, prob)))

        # Gradient
        grad = [0.0] * p
        for j in range(p):
            grad[j] = sum(X[i][j] * (y[i] - pi[i]) for i in range(n))

        # Hessian = - (X^T W X)
        H = [[0.0] * p for _ in range(p)]
        for j in range(p):
            for k in range(p):
                H[j][k] = sum(X[i][j] * X[i][k] * pi[i] * (1.0 - pi[i]) for i in range(n))

        # Invert 4x4 matrix using standard gauss-jordan
        inv_H = [row[:] for row in H]
        identity = [[1.0 if r == c else 0.0 for c in range(p)] for r in range(p)]
        for i in range(p):
            inv_H[i].extend(identity[i])

        for i in range(p):
            pivot = inv_H[i][i]
            if abs(pivot) < 1e-14:
                pivot = 1e-14
            for c in range(2 * p):
                inv_H[i][c] /= pivot
            for r in range(p):
                if r != i:
                    factor = inv_H[r][i]
                    for c in range(2 * p):
                        inv_H[r][c] -= factor * inv_H[i][c]

        inv_H_final = [row[p:] for row in inv_H]

        # Update step: delta = inv_H * grad
        delta = [sum(inv_H_final[j][k] * grad[k] for k in range(p)) for j in range(p)]
        for j in range(p):
            beta[j] += delta[j]

        if max(abs(d) for d in delta) < 1e-7:
            break

    # Standard errors
    se = [math.sqrt(max(0.0, inv_H_final[j][j])) for j in range(p)]
    z_scores = [beta[j] / se[j] if se[j] > 0 else 0.0 for j in range(p)]
    p_values = [2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0)))) for z in z_scores]

    terms = ["Intercept", "Code_Exposure", "Model_Diversity", "Exposure_x_Diversity"]
    model_res = {}
    for j in range(p):
        or_val = math.exp(beta[j])
        ci_lower = math.exp(beta[j] - 1.96 * se[j])
        ci_upper = math.exp(beta[j] + 1.96 * se[j])
        model_res[terms[j]] = {
            "coef": round(beta[j], 4),
            "se": round(se[j], 4),
            "z": round(z_scores[j], 3),
            "p_value": round(p_values[j], 5),
            "odds_ratio": round(or_val, 4),
            "ci_95": (round(ci_lower, 4), round(ci_upper, 4))
        }

    return {"n": n, "coefficients": model_res}


def run_causal_mediation_analysis(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Formal Causal Mediation Analysis (Imai et al. / Pearl Counterfactuals).
    Treatment T: Code Exposure (0 = Spec-only, 1 = Code-first)
    Mediator M: Suite Usability (0 = Unusable, 1 = Usable)
    Outcome Y: Defect Detection (0 = Not detected, 1 = Detected)
    Computes:
      - Total Effect (TE)
      - Average Causal Mediation Effect (ACME / NIE)
      - Natural Direct Effect (NDE)
      - Proportion of Total Effect Mediated by Usability (PM = ACME / TE)
    """
    # Filter to main comparison (Same model: T0 vs T1)
    sub = [r for r in records if r["condition"] in ("T0_Spec_Same", "T1_Code_Same")]
    
    # Stratified empirical expectations:
    # P(M=1 | T=1), P(M=1 | T=0)
    m_t1 = [1.0 if r["is_usable"] == "True" else 0.0 for r in sub if r["condition"] == "T1_Code_Same"]
    m_t0 = [1.0 if r["is_usable"] == "True" else 0.0 for r in sub if r["condition"] == "T0_Spec_Same"]
    
    p_m1_given_t1 = sum(m_t1) / len(m_t1)
    p_m1_given_t0 = sum(m_t0) / len(m_t0)
    
    # P(Y=1 | T=t, M=m)
    y_t1_m1 = [1.0 if r["detects_defect"] == "True" else 0.0 for r in sub if r["condition"] == "T1_Code_Same" and r["is_usable"] == "True"]
    y_t1_m0 = [1.0 if r["detects_defect"] == "True" else 0.0 for r in sub if r["condition"] == "T1_Code_Same" and r["is_usable"] == "False"]
    y_t0_m1 = [1.0 if r["detects_defect"] == "True" else 0.0 for r in sub if r["condition"] == "T0_Spec_Same" and r["is_usable"] == "True"]
    y_t0_m0 = [1.0 if r["detects_defect"] == "True" else 0.0 for r in sub if r["condition"] == "T0_Spec_Same" and r["is_usable"] == "False"]
    
    e_y_t1_m1 = sum(y_t1_m1) / len(y_t1_m1) if y_t1_m1 else 0.0
    e_y_t1_m0 = sum(y_t1_m0) / len(y_t1_m0) if y_t1_m0 else 0.0
    e_y_t0_m1 = sum(y_t0_m1) / len(y_t0_m1) if y_t0_m1 else 0.0
    e_y_t0_m0 = sum(y_t0_m0) / len(y_t0_m0) if y_t0_m0 else 0.0
    
    # Counterfactual outcome expectations:
    # E[Y(1, M(1))] = E[Y | T=1]
    e_y1_m1 = e_y_t1_m1 * p_m1_given_t1 + e_y_t1_m0 * (1.0 - p_m1_given_t1)
    # E[Y(0, M(0))] = E[Y | T=0]
    e_y0_m0 = e_y_t0_m1 * p_m1_given_t0 + e_y_t0_m0 * (1.0 - p_m1_given_t0)
    # E[Y(1, M(0))] = Potential outcome under Treatment=1, with mediator value set to what it would be under Control=0
    e_y1_m0 = e_y_t1_m1 * p_m1_given_t0 + e_y_t1_m0 * (1.0 - p_m1_given_t0)
    
    # Causal Effects:
    # Total Effect = E[Y(1, M(1))] - E[Y(0, M(0))]
    total_effect = e_y1_m1 - e_y0_m0
    # Natural Direct Effect (NDE) = E[Y(1, M(0))] - E[Y(0, M(0))]
    nde = e_y1_m0 - e_y0_m0
    # Average Causal Mediation Effect (ACME) = E[Y(1, M(1))] - E[Y(1, M(0))]
    acme = e_y1_m1 - e_y1_m0
    
    proportion_mediated = (acme / total_effect) if abs(total_effect) > 1e-6 else 0.0
    
    return {
        "p_usable_given_spec": round(p_m1_given_t0, 4),
        "p_usable_given_code": round(p_m1_given_t1, 4),
        "delta_usability": round(p_m1_given_t1 - p_m1_given_t0, 4),
        "e_y_t1": round(e_y1_m1, 4),
        "e_y_t0": round(e_y0_m0, 4),
        "total_effect": round(total_effect, 4),
        "natural_direct_effect": round(nde, 4),
        "acme_indirect_effect": round(acme, 4),
        "proportion_mediated_pct": round(proportion_mediated * 100, 2)
    }


def main():
    records = load_dataset()
    print(f"Loaded {len(records)} records.")
    assert len(records) == 900, f"Expected 900 records, found {len(records)}"
    
    # 1. Descriptive
    desc = run_descriptive_analysis(records)
    print("\n=== DESCRIPTIVE STATISTICS (N=225 per condition) ===")
    for c, s in desc.items():
        print(f"{c:15s} | Usable: {s['n_usable']}/225 ({s['usable_pct']:.2f}%) | Raw Detect: {s['n_detect']}/225 ({s['detect_pct']:.2f}%) | Cond Detect: {s['cond_detect_pct']:.2f}%")

    # 2. Paired tests
    paired = run_paired_comparisons(records)
    print("\n=== PAIRED COMPARISONS (N=225 Matched Quadruplets) ===")
    for k, p in paired.items():
        mcn = p["mcnemar"]
        tost = p["tost"]
        print(f"\n--- {p['description']} ({k}) ---")
        print(f"  Ref: {p['ref_rate']:.2f}% vs Focal: {p['focal_rate']:.2f}% (Delta = {p['delta_pp']:+.2f} pp)")
        print(f"  Exact McNemar: p = {mcn['p_value_exact_binomial']:.2e} ({mcn['p_value_exact_binomial']:.6f}) | Discordant: (Ref-only={mcn['b_armA_only']}, Focal-only={mcn['c_armB_only']})")
        print(f"  Odds Ratio: {mcn['odds_ratio']} (95% CI: {mcn['ci_95']})")
        print(f"  TOST Equivalence (margin=8%): p_TOST = {tost['p_tost']:.6f} | Equivalent: {tost['is_equivalent']}")

    # 3. Logistic Model
    model = fit_logistic_model(records)
    print("\n=== 2x2 FACTORIAL LOGISTIC REGRESSION WITH INTERACTION ===")
    for term, vals in model["coefficients"].items():
        print(f"{term:22s} | beta={vals['coef']:+7.4f} | SE={vals['se']:.4f} | z={vals['z']:+6.3f} | p={vals['p_value']:.5f} | OR={vals['odds_ratio']:.4f} (95% CI: {vals['ci_95']})")

    # 4. Causal Mediation
    med = run_causal_mediation_analysis(records)
    print("\n=== CAUSAL MEDIATION ANALYSIS (ACME) ===")
    print(f"  Total Effect (TE):                    {med['total_effect']:+.4f} ({med['total_effect']*100:+.2f} pp)")
    print(f"  Natural Direct Effect (NDE):          {med['natural_direct_effect']:+.4f} ({med['natural_direct_effect']*100:+.2f} pp)")
    print(f"  Average Causal Mediation Effect (ACME): {med['acme_indirect_effect']:+.4f} ({med['acme_indirect_effect']*100:+.2f} pp)")
    print(f"  Proportion Mediated by Usability:     {med['proportion_mediated_pct']:.2f}%")

    # Export consolidated summary to json
    results_payload = {
        "descriptive": desc,
        "paired": paired,
        "logistic_model": model,
        "causal_mediation": med
    }
    out_file = project_root / "results" / "complete_statistical_analysis.json"
    out_file.write_text(json.dumps(results_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAll complete statistical analyses saved to: {out_file}")


if __name__ == "__main__":
    main()
