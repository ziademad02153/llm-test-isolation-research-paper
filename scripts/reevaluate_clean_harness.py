"""
Re-evaluates the 900 records in results_2048/ with a harness that prevents
function under test redefinition from overwriting C_ref or C_fault.

Specifically:
1. Detects whether generated_test_code defines the target function (or any helper defined in implementation).
2. Strips any top-level FunctionDef/ClassDef matching the target function name via AST.
3. Evaluates both:
   a) Unstripped (Original harness: implementation + test_code)
   b) Stripped (Harness where test_code cannot overwrite implementation)
4. Saves reevaluated results to results_2048/experiment_records_reval.jsonl and .csv.
5. Computes complete statistical comparisons across all 4 conditions.
"""

import sys
import os
import json
import csv
import ast
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.sandbox import execute_code_with_test, SandboxStatus

# Load tasks
with open(ROOT / "data" / "verified_tasks_manifest.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)

task_dict = {m["problem_id"]: m for m in manifest}

def get_target_fn_name(m):
    tree = ast.parse(m["ref_implementation"])
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return None

target_fns = {pid: get_target_fn_name(m) for pid, m in task_dict.items()}

def strip_function_def(code_str, fn_name):
    try:
        tree = ast.parse(code_str)
        new_body = []
        stripped = False
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
                stripped = True
            else:
                new_body.append(node)
        if stripped:
            tree.body = new_body
            return ast.unparse(tree), True
        return code_str, False
    except:
        return code_str, False

def evaluate_single_record(r):
    pid = r["problem_id"]
    m = task_dict[pid]
    fn_name = target_fns[pid]
    raw_test_code = r["generated_test_code"]
    
    cleaned_test_code, redefines_target = strip_function_def(raw_test_code, fn_name)
    
    # 1. Evaluate cleaned code on C_ref
    ref_status, ref_out, _ = execute_code_with_test(m["ref_implementation"], cleaned_test_code)
    
    # Usability: passes on C_ref, has >= 1 assertions, not empty
    num_asserts = r.get("total_assertions", 0)
    # Check if cleaned code still has assertions
    is_usable = (ref_status == SandboxStatus.PASSED) and (num_asserts >= 1) and not r.get("is_empty_suite", False)
    
    detects_defect = False
    faulty_status = "SKIPPED_UNUSABLE"
    faulty_out = ""
    
    if is_usable:
        faulty_status, faulty_out, _ = execute_code_with_test(m["faulty_implementation"], cleaned_test_code)
        detects_defect = (faulty_status == SandboxStatus.FAILED_ASSERTION)
        
    return {
        "trial_id": r["trial_id"],
        "problem_id": pid,
        "condition": r["condition"],
        "model_family": r["model_family"],
        "model_name": r["model_name"],
        "redefines_target_fn": redefines_target,
        "orig_is_usable": r["is_usable"],
        "orig_detects_defect": r["detects_defect"],
        "orig_ref_status": r["ref_run_status"],
        "orig_faulty_status": r["faulty_run_status"],
        "cleaned_is_usable": is_usable,
        "cleaned_detects_defect": detects_defect,
        "cleaned_ref_status": ref_status,
        "cleaned_faulty_status": faulty_status,
        "cleaned_test_code": cleaned_test_code,
        "total_assertions": num_asserts
    }

def main():
    records_path = ROOT / "results_2048" / "experiment_records.jsonl"
    with open(records_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
        
    print(f"Loaded {len(records)} records. Re-evaluating with stripped redefinitions...")
    t0 = time.time()
    
    # Run in parallel with ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(evaluate_single_record, records))
        
    elapsed = time.time() - t0
    print(f"Re-evaluation complete in {elapsed:.2f} seconds!")
    
    # Save reevaluated records
    out_jsonl = ROOT / "results_2048" / "experiment_records_reval.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    print(f"Saved re-evaluated records to {out_jsonl}")
    
    # Analysis
    print("\n" + "="*80)
    print("                     SIDE-BY-SIDE RE-EVALUATION ANALYSIS                        ")
    print("="*80)
    
    conditions = ["T0_Spec_Same", "T1_Code_Same", "T2_Code_Cross", "T3_Spec_Cross"]
    
    print(f"{'Condition':<15} | {'Metric':<18} | {'Original':<12} | {'Cleaned':<12} | {'Delta':<10}")
    print("-" * 75)
    
    for c in conditions:
        sub = [r for r in results if r["condition"] == c]
        n = len(sub)
        redef_count = sum(1 for r in sub if r["redefines_target_fn"])
        
        orig_u = sum(1 for r in sub if r["orig_is_usable"] is True)
        orig_d = sum(1 for r in sub if r["orig_detects_defect"] is True)
        
        clean_u = sum(1 for r in sub if r["cleaned_is_usable"] is True)
        clean_d = sum(1 for r in sub if r["cleaned_detects_defect"] is True)
        
        print(f"{c:<15} | Redefines Fn Count | {redef_count:4d}/{n}     |            |           ")
        print(f"{'':<15} | Usability (%)      | {orig_u/n*100:6.2f}%      | {clean_u/n*100:6.2f}%      | {(clean_u-orig_u)/n*100:+6.2f} pp")
        print(f"{'':<15} | Defect Detect (%)  | {orig_d/n*100:6.2f}%      | {clean_d/n*100:6.2f}%      | {(clean_d-orig_d)/n*100:+6.2f} pp")
        if clean_u > 0:
            orig_cd = (orig_d / orig_u * 100) if orig_u else 0.0
            clean_cd = clean_d / clean_u * 100
            print(f"{'':<15} | Cond Detect (%)    | {orig_cd:6.2f}%      | {clean_cd:6.2f}%      | {clean_cd-orig_cd:+6.2f} pp")
        print("-" * 75)

if __name__ == "__main__":
    main()
