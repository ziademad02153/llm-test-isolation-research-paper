"""
Data schema definitions for the Causal Empirical Study in Software Engineering:
Disentangling Code Exposure from Model Diversity & Oracle Validity Breakdown.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum


class ExperimentalCondition(str, Enum):
    T0_SPEC_SAME = "T0_Spec_Same"      # Specification-only, same model family
    T1_CODE_SAME = "T1_Code_Same"      # Code-first (implementation visible), same model family
    T2_CODE_CROSS = "T2_Code_Cross"    # Code-first (implementation visible), cross-family model
    T3_SPEC_CROSS = "T3_Spec_Cross"    # Specification-only, cross-family model


@dataclass
class ProblemSpec:
    problem_id: str
    function_name: str
    signature: str
    docstring: str
    ref_implementation: str
    golden_test: str
    category: str
    difficulty: str
    faulty_implementation: Optional[str] = None
    fault_type: Optional[str] = None
    fault_generating_model: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationRecord:
    trial_id: str
    problem_id: str
    condition: str
    model_family: str
    model_name: str
    is_spec_only: bool
    is_cross_family: bool
    generated_test_code: str
    total_assertions: int
    is_usable: bool                  # Level 1: runs on ref_implementation without error and has >=1 assertion
    detects_defect: bool             # Level 2: fails on faulty_implementation (only evaluated if usable)
    ref_run_status: str              # PASSED, FAILED_ASSERTION, RUNTIME_EXCEPTION, TIMEOUT, EMPTY_ORACLE
    faulty_run_status: str           # PASSED, FAILED_ASSERTION, RUNTIME_EXCEPTION, TIMEOUT, SKIPPED_UNUSABLE
    ref_output: str = ""
    faulty_output: str = ""
    execution_time_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    raw_completion: str = ""
    has_syntax_error: bool = False
    is_empty_suite: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

