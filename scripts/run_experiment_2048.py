"""
Hardened 2x2 Factorial Causal Experiment Runner (2048-Token Execution).
Key Enhancements:
  1. max_tokens = 2048 (eliminates token budget truncation ceiling).
  2. Preserves raw completion text, explicit finish_reason, and syntax error flags.
  3. Strict Level 1 Usability: requires passing C_ref AND having at least 1 verified assertion.
  4. Randomized trial schedule across all 900 trials (seed=42) to eliminate temporal order confounding.
  5. Dedicated output directory: results_2048/
  6. Robust crash-proof resume capability.
"""

import os
import sys
import json
import time
import csv
import random
import argparse
from pathlib import Path
from typing import Dict, Any, List, Set

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.schema import ProblemSpec, EvaluationRecord, ExperimentalCondition
from src.sandbox import execute_code_with_test, SandboxStatus
from src.runner import GroqPoolClient, extract_python_code, count_assert_statements, PROMPT_SPEC_ONLY, PROMPT_CODE_FIRST

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


def evaluate_single_trial_2048(
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

    # 2. Call LLM with max_tokens = 2048
    start_t = time.perf_counter()
    content, usage, finish_reason, raw_text = client.generate(
        model=model_name,
        messages=messages,
        temperature=0.7,
        max_tokens=2048
    )
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    test_code, is_empty_suite, has_syntax_error = extract_python_code(content)
    num_asserts = count_assert_statements(test_code)

    # 3. Level 1: Run against ref_implementation (Oracle Usability)
    if is_empty_suite:
        ref_status = "EMPTY_SUITE"
        ref_out = ""
        is_usable = False
    elif has_syntax_error:
        ref_status = SandboxStatus.SYNTAX_ERROR
        ref_out = "SyntaxError in generated test code"
        is_usable = False
    else:
        ref_status, ref_out, _ = execute_code_with_test(prob.ref_implementation, test_code)
        # Strict oracle usability requires zero execution errors AND >= 1 assertion
        if ref_status == SandboxStatus.PASSED:
            if num_asserts >= 1:
                is_usable = True
            else:
                is_usable = False
                ref_status = "EMPTY_ORACLE"  # Ran cleanly but verified 0 assertions
        else:
            is_usable = False

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
        completion_tokens=usage.get("completion_tokens", 0),
        finish_reason=finish_reason,
        raw_completion=raw_text,
        has_syntax_error=has_syntax_error,
        is_empty_suite=is_empty_suite
    )


def run_experiment_2048(repetitions: int = 15, pilot: bool = False, force: bool = False):
    problems = load_all_problems()
    if pilot:
        problems = problems[:2]
        repetitions = 1
        print(f"--- RUNNING PILOT 2048: {len(problems)} tasks, {repetitions} rep (Total {len(problems)*4*repetitions} evals) ---")
    else:
        print(f"--- RUNNING FULL 2048 EXPERIMENT: {len(problems)} tasks, {repetitions} reps (Total {len(problems)*4*repetitions} evals) ---")

    client = GroqPoolClient()
    results_dir = project_root / "results_2048"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_file = results_dir / "experiment_records.csv"
    jsonl_file = results_dir / "experiment_records.jsonl"

    fieldnames = [
        "trial_id", "problem_id", "condition", "model_family", "model_name",
        "is_spec_only", "is_cross_family", "total_assertions",
        "is_usable", "detects_defect", "ref_run_status", "faulty_run_status",
        "execution_time_ms", "prompt_tokens", "completion_tokens",
        "finish_reason", "has_syntax_error", "is_empty_suite"
    ]

    # Check already completed trials for resuming
    completed_ids: Set[str] = set()
    if jsonl_file.exists():
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        completed_ids.add(obj["trial_id"])
                    except Exception:
                        pass
        print(f"Found {len(completed_ids)} already completed trials in {jsonl_file.name}. Resuming...")

    # Build full trial list
    trial_queue = []
    for prob in problems:
        for cond, model_name, is_spec_only, is_cross in CONDITIONS:
            for r in range(1, repetitions + 1):
                trial_id = f"{prob.problem_id}_{cond.value}_R{r:02d}"
                if trial_id not in completed_ids:
                    trial_queue.append((prob, cond, model_name, is_spec_only, is_cross, r, trial_id))

    # Randomize trial queue to eliminate temporal order confounding
    random.seed(42)
    random.shuffle(trial_queue)

    print(f"Pending evaluations to execute: {len(trial_queue)} / {len(problems)*4*repetitions}")
    if not trial_queue:
        print("All trials already completed! Nothing to run.")
        return

    # Open CSV writer in append mode
    csv_exists = csv_file.exists() and csv_file.stat().st_size > 0
    csv_f = open(csv_file, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_f, fieldnames=fieldnames, extrasaction="ignore")
    if not csv_exists:
        writer.writeheader()
        csv_f.flush()

    jsonl_f = open(jsonl_file, "a", encoding="utf-8")

    start_total_t = time.perf_counter()
    done_count = len(completed_ids)
    total_planned = len(problems) * 4 * repetitions

    try:
        for idx, (prob, cond, model_name, is_spec_only, is_cross, r, trial_id) in enumerate(trial_queue, 1):
            record = evaluate_single_trial_2048(
                client=client,
                prob=prob,
                cond=cond,
                model_name=model_name,
                is_spec_only=is_spec_only,
                is_cross=is_cross,
                trial_num=r
            )

            # Write immediately to disk
            writer.writerow(record.to_dict())
            csv_f.flush()

            jsonl_f.write(json.dumps(record.to_dict()) + "\n")
            jsonl_f.flush()

            done_count += 1
            if idx % 5 == 0 or idx == len(trial_queue):
                elapsed_mins = (time.perf_counter() - start_total_t) / 60.0
                rate_per_min = idx / elapsed_mins if elapsed_mins > 0 else 0
                rem_trials = len(trial_queue) - idx
                eta_mins = rem_trials / rate_per_min if rate_per_min > 0 else 0
                print(
                    f"[{done_count}/{total_planned} | {done_count/total_planned*100:5.1f}%] "
                    f"{trial_id:30s} | Usable={str(record.is_usable):5s} | Detect={str(record.detects_defect):5s} | "
                    f"Finish={record.finish_reason:7s} | Asserts={record.total_assertions:2d} | ETA: {eta_mins:.1f}m"
                )

            # Polite pacing between API calls
            time.sleep(0.75)

    finally:
        csv_f.close()
        jsonl_f.close()

    total_mins = (time.perf_counter() - start_total_t) / 60.0
    print(f"\n2048-TOKEN EXPERIMENT COMPLETED in {total_mins:.2f} minutes!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run quick 2-task 1-trial pilot")
    parser.add_argument("--repetitions", type=int, default=15, help="Number of repetitions per condition")
    parser.add_argument("--force", action="store_true", help="Force run")
    args = parser.parse_args()
    run_experiment_2048(repetitions=args.repetitions, pilot=args.pilot, force=args.force)
