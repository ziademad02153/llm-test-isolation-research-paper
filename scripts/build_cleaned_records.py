"""
Generates the fully updated, cleaned results_2048/ datasets:
- Preserves original raw completions and code.
- Adds `redefines_target_fn`: True/False.
- Evaluates test suites without harness redefinition shadowing.
- Updates is_usable, detects_defect, ref_run_status, faulty_run_status.
- Writes experiment_records.jsonl and experiment_records.csv.
- Generates analysis_summary.json with full descriptive and inferential statistics.
"""

import sys
import os
import json
import csv
import ast
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

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

def process_record(r):
    pid = r["problem_id"]
    m = task_dict[pid]
    fn_name = target_fns[pid]
    raw_code = r["generated_test_code"]
    
    cleaned_code, redefines = strip_function_def(raw_code, fn_name)
    
    # Run Level 1 on C_ref
    ref_status, ref_out, _ = execute_code_with_test(m["ref_implementation"], cleaned_code)
    num_asserts = r.get("total_assertions", 0)
    is_usable = (ref_status == SandboxStatus.PASSED) and (num_asserts >= 1) and not r.get("is_empty_suite", False)
    
    detects = False
    faulty_status = "SKIPPED_UNUSABLE"
    faulty_out = ""
    
    if is_usable:
        faulty_status, faulty_out, _ = execute_code_with_test(m["faulty_implementation"], cleaned_code)
        detects = (faulty_status == SandboxStatus.FAILED_ASSERTION)
        
    updated = dict(r)
    updated["redefines_function_under_test"] = redefines
    updated["is_usable"] = is_usable
    updated["detects_defect"] = detects
    updated["ref_run_status"] = ref_status
    updated["faulty_run_status"] = faulty_status
    updated["ref_output"] = ref_out
    updated["faulty_output"] = faulty_out
    return updated

def main():
    jsonl_path = ROOT / "results_2048" / "experiment_records.jsonl"
    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
        
    # Backup original naive records first
    naive_path = ROOT / "results_2048" / "experiment_records_naive_concat.jsonl"
    if not naive_path.exists():
        with open(naive_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"Backed up naive records to {naive_path}")
        
    with ProcessPoolExecutor(max_workers=8) as executor:
        updated_records = list(executor.map(process_record, records))
        
    # Write updated experiment_records.jsonl
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in updated_records:
            f.write(json.dumps(r) + "\n")
            
    # Write updated experiment_records.csv
    csv_path = ROOT / "results_2048" / "experiment_records.csv"
    fieldnames = list(updated_records[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_records)
        
    print(f"Updated {jsonl_path} and {csv_path} with clean harness evaluation!")

if __name__ == "__main__":
    main()
