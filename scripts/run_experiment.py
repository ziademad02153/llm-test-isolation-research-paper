"""
Main 2x2 Factorial Causal Experiment Runner.
Executes 900 test generation evaluations across 15 tasks (15 quadruplets x 4 conditions).
Evaluates Two-Level Oracle Metrics:
  Level 1: Suite Usability (Runs on ref_implementation without error).
  Level 2: Defect Detection (Fails on faulty_implementation among usable suites).
"""

import os
import sys
import json
import time
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.schema import ProblemSpec, EvaluationRecord, ExperimentalCondition
from src.sandbox import execute_code_with_test, SandboxStatus
from src.runner import GroqPoolClient, extract_python_code, PROMPT_SPEC_ONLY, PROMPT_CODE_FIRST

MODEL_SAME = "openai/gpt-oss-120b"
MODEL_CROSS = "qwen/qwen3.8-27b"

CONDITIONS = [
    (ExperimentalCondition.T0_SPEC_SAME, MODEL_SAME, True, False),
    (ExperimentalCondition.T1_CODE_SAME, MODEL_SAME, False, False),
    (ExperimentalCondition.T2_CODE_CROSS, MODEL_CROSS, False, True),
    (ExperimentalCondition.T3_SPEC_CROSS, MODEL_CROSS, True, True),
]


def load_all_problems() -> List[ProblemSpec]:
    problems_dir = project_root / "data" / "problems"
    files = sorted(problems_dir.glob("TASK_*.json"))
    problems = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        problems.append(ProblemSpec(**data))
    return problems


def count_assert_statements(code: str) -> int:
    return len([line for line in code.splitlines() if line.strip().startswith("assert ")])


def evaluate_single_trial(
    client: GroqPoolClient,
    prob: ProblemSpec,
    cond: ExperimentalCondition,
    model_name: str,
    is_spec_only: bool,
    is_cross: bool,
    trial_num: int
) -> EvaluationRecord:
    trial_id = f"{prob.problem_id}_{cond.value}_R{trial_num:02d}"
    
    # 1. Build prompt
    if is_spec_only:
        prompt_text = PROMPT_SPEC_ONLY.format(
            signature=prob.signature,
            docstring=prob.docstring
        )
    else:
        prompt_text = PROMPT_CODE_FIRST.format(
            signature=prob.signature,
            docstring=prob.docstring,
            implementation=prob.faulty_implementation
        )
        
    messages = [{"role": "user", "content": prompt_text}]
    
    # 2. Call LLM
    start_t = time.perf_counter()
    raw_response, usage = client.generate(
        model=model_name,
        messages=messages,
        temperature=0.7,
        max_tokens=512
    )
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0
    
    test_code = extract_python_code(raw_response)
    num_asserts = count_assert_statements(test_code)
    
    # Pace requests slightly to respect token burst limits
    time.sleep(2.5)
    
    # 3. Level 1: Run against ref_implementation (Oracle Usability)
    ref_status, ref_out, _ = execute_code_with_test(prob.ref_implementation, test_code)
    is_usable = (ref_status == SandboxStatus.PASSED)
    
    # 4. Level 2: Run against faulty_implementation (Defect Detection)
    detects_defect = False
    faulty_status = "SKIPPED_UNUSABLE"
    faulty_out = ""
    
    if is_usable:
        faulty_status, faulty_out, _ = execute_code_with_test(prob.faulty_implementation, test_code)
        detects_defect = (faulty_status == SandboxStatus.FAILED_ASSERTION)
        
    family = "OpenAI" if "openai" in model_name else "Qwen"
    
    return EvaluationRecord(
        trial_id=trial_id,
        problem_id=prob.problem_id,
        condition=cond.value,
        model_family=family,
        model_name=model_name,
        is_spec_only=is_spec_only,
        is_cross_family=is_cross,
        generated_test_code=test_code,
        total_assertions=num_asserts,
        is_usable=is_usable,
        detects_defect=detects_defect,
        ref_run_status=ref_status,
        faulty_run_status=faulty_status,
        ref_output=ref_out[:300],
        faulty_output=faulty_out[:300],
        execution_time_ms=elapsed_ms,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0)
    )


def run_experiment(repetitions: int = 15, pilot: bool = False, force: bool = False):
    problems = load_all_problems()
    if pilot:
        problems = problems[:2]
        repetitions = 1
        print(f"--- RUNNING PILOT: {len(problems)} tasks, {repetitions} repetition (Total {len(problems)*4*repetitions} evaluations) ---")
    else:
        print(f"--- RUNNING FULL EXPERIMENT: {len(problems)} tasks, {repetitions} repetitions (Total {len(problems)*4*repetitions} evaluations) ---")

    client = GroqPoolClient()
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = results_dir / "experiment_records.csv"
    jsonl_file = results_dir / "experiment_records.jsonl"

    if not force and not pilot and csv_file.exists() and csv_file.stat().st_size > 0:
        raise SystemExit(
            "SAFETY GUARD: results/experiment_records.csv already exists and is not empty.\n"
            "Executing live inference will overwrite recorded empirical traces.\n"
            "To confirm re-execution, pass the --force flag:\n"
            "    python scripts/run_experiment.py --repetitions 15 --force"
        )
    
    fieldnames = [
        "trial_id", "problem_id", "condition", "model_family", "model_name",
        "is_spec_only", "is_cross_family", "total_assertions",
        "is_usable", "detects_defect", "ref_run_status", "faulty_run_status",
        "execution_time_ms", "prompt_tokens", "completion_tokens"
    ]
    
    # Check existing progress for resume capability
    existing_trials = set()
    if csv_file.exists():
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_trials.add(row["trial_id"])
        print(f"Resuming: {len(existing_trials)} trials already completed.")
    else:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    total_evals = len(problems) * 4 * repetitions
    done_count = len(existing_trials)
    
    start_total_t = time.perf_counter()
    
    for r in range(1, repetitions + 1):
        for prob in problems:
            for cond, model, spec_only, cross in CONDITIONS:
                trial_id = f"{prob.problem_id}_{cond.value}_R{r:02d}"
                if trial_id in existing_trials:
                    continue
                    
                rec = evaluate_single_trial(client, prob, cond, model, spec_only, cross, r)
                
                # Append to CSV
                with open(csv_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    row = rec.to_dict()
                    del row["generated_test_code"]
                    del row["ref_output"]
                    del row["faulty_output"]
                    writer.writerow(row)
                    
                # Append to JSONL
                with open(jsonl_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
                    
                existing_trials.add(trial_id)
                done_count += 1
                
                print(f"[{done_count}/{total_evals}] {trial_id} | Model={rec.model_name.split('/')[-1]} | Usable={rec.is_usable} | Detects={rec.detects_defect} ({rec.execution_time_ms:.0f}ms)")

    total_mins = (time.perf_counter() - start_total_t) / 60.0
    print(f"\nEXPERIMENT COMPLETED in {total_mins:.2f} minutes!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run quick 2-task 1-trial pilot")
    parser.add_argument("--repetitions", type=int, default=15, help="Number of repetitions per condition")
    parser.add_argument("--force", action="store_true", help="Confirm overwrite of existing empirical records")
    args = parser.parse_args()
    run_experiment(repetitions=args.repetitions, pilot=args.pilot, force=args.force)
