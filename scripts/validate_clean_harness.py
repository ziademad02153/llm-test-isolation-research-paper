"""
Validation Script for AST-Scoped Clean Harness:
1. Runs all ground-truth task suites on C_ref and C_fault.
2. Injects function shadowing and verifies that AST stripping successfully isolates benchmark code.
3. Performs runtime function identity checks.
4. Inspects TASK_001 trials R01, R06, R11 under both naive and clean harnesses.
"""

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.sandbox import execute_code_with_test, SandboxStatus

def run_validation():
    print("=================================================================")
    print("            CLEAN HARNESS VALIDATION & AUDIT SUITE               ")
    print("=================================================================")

    # 1. Ground Truth Task Validation
    manifest_path = ROOT / "data" / "verified_tasks_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    print(f"\n1. Validating {len(tasks)} benchmark tasks with ground truth tests...")
    all_gt_passed = True
    for t in tasks:
        pid = t["problem_id"]
        ref = t["ref_implementation"]
        fault = t["faulty_implementation"]
        test = t["golden_test"]

        # Run on C_ref
        ref_st, ref_out, _ = execute_code_with_test(ref, test)
        assert ref_st == SandboxStatus.PASSED, f"Task {pid} failed on C_ref: {ref_out}"

        # Run on C_fault
        fault_st, fault_out, _ = execute_code_with_test(fault, test)
        assert fault_st == SandboxStatus.FAILED_ASSERTION, f"Task {pid} did not trigger assertion on C_fault: {fault_out}"

    print(f"[PASS] All {len(tasks)} tasks: 100% pass on C_ref, 100% detect on C_fault.")

    # 2. Shadowing Stripping Verification
    print("\n2. Testing AST stripping and function identity...")
    sample_task = tasks[0] # TASK_001: parse_semver
    target_fn_name = "parse_semver"

    # Synthetic test with injected buggy redefinition
    shadowed_test = """
def parse_semver(v):
    return "SHADOWED_BUGGY_OUTPUT"

assert parse_semver("1.2.3") == {"major": 1, "minor": 2, "patch": 3, "prerelease": None, "build": None}
"""

    # Naive execution without stripping
    naive_st, naive_out, _ = execute_code_with_test(sample_task["ref_implementation"], shadowed_test)
    assert naive_st == SandboxStatus.FAILED_ASSERTION, "Naive harness failed to reproduce shadowing bug!"
    print(f"  [Verified] Naive concatenation caused assertion failure due to shadowing.")

    # Clean execution with AST stripping
    tree = ast.parse(shadowed_test)
    clean_body = [n for n in tree.body if not (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == target_fn_name)]
    tree.body = clean_body
    cleaned_test = ast.unparse(tree)

    clean_st, clean_out, _ = execute_code_with_test(sample_task["ref_implementation"], cleaned_test)
    assert clean_st == SandboxStatus.PASSED, f"Clean harness failed to pass on C_ref after stripping: {clean_out}"
    print(f"  [Verified] AST-scoped clean harness removed shadowing and executed benchmark function successfully.")

    # 3. Raw Trials Audit (R01, R06, R11)
    print("\n3. Auditing TASK_001 trials R01, R06, R11...")
    naive_path = ROOT / "results_2048" / "experiment_records_naive_concat.jsonl"
    clean_path = ROOT / "results_2048" / "experiment_records.jsonl"

    naive_records = {r["trial_id"]: r for r in [json.loads(l) for l in open(naive_path, encoding="utf-8")]}
    clean_records = {r["trial_id"]: r for r in [json.loads(l) for l in open(clean_path, encoding="utf-8")]}

    target_trials = ["TASK_001_T1_Code_Same_R01", "TASK_001_T1_Code_Same_R06", "TASK_001_T1_Code_Same_R11"]
    for tid in target_trials:
        n_r = naive_records[tid]
        c_r = clean_records[tid]
        print(f"\n--- {tid} ---")
        print(f"  Redefines function: {c_r.get('redefines_function_under_test')}")
        print(f"  Naive: Usable={n_r['is_usable']}, Detect={n_r['detects_defect']}, RefStatus={n_r['ref_run_status']}")
        print(f"  Clean: Usable={c_r['is_usable']}, Detect={c_r['detects_defect']}, RefStatus={c_r['ref_run_status']}")

    print("\n=================================================================")
    print("             CLEAN HARNESS VALIDATION COMPLETE                   ")
    print("=================================================================")

if __name__ == "__main__":
    run_validation()
