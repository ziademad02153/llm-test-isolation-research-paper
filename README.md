# Replication Package

## Separating Test Synthesis from Implementation Inspection: An Empirical Factorial Study of Oracle Validity, Code Exposure, and Test Harness Scoping in LLM Testing

- **Author:** Ziad Emad Allam (ORCID: [0009-0008-4323-791X](https://orcid.org/0009-0008-4323-791X))
- **Affiliation:** The American University in Cairo (AUC), New Cairo, Egypt
- **Contact:** `z.e.ziad@xed.aucegypt.edu`
- **License:** MIT
- **Python Version:** 3.10+ (Standard Library Only)
- **Study Type:** Empirical Factorial Software Engineering Experiment

This repository contains the complete replication package for the empirical research study investigating the impact of **Code Exposure** (Specification-Only vs. Code-First) and **Test-Generation Model** (`openai/gpt-oss-120b` vs. `qwen/qwen3.8-27b`) on LLM-generated test suite efficacy, test oracle validity, defect detection, and **Test Execution Harness Scoping Architectures**.

---

## 1. Overview & Experimental Design

The study follows a **balanced 2 x 2 factorial experiment** across 15 curated software logic defect tasks (225 trials per condition, 15 tasks x 15 repetitions, 900 recorded condition evaluations total under an unconstrained 2048-token budget):

| Condition | Code Exposure | Test-Generation Model | Prompt Content |
| :--- | :--- | :--- | :--- |
| **T0 (Spec-GPT)** | No (Specification Only) | `openai/gpt-oss-120b` | Function signature + natural language docstring |
| **T1 (Code-GPT)** | Yes (Code-First Visible) | `openai/gpt-oss-120b` | Signature + docstring + buggy source code (C_fault) |
| **T2 (Code-Qwen)** | Yes (Code-First Visible) | `qwen/qwen3.8-27b` | Signature + docstring + buggy source code (C_fault) |
| **T3 (Spec-Qwen)** | No (Specification Only) | `qwen/qwen3.8-27b` | Function signature + natural language docstring |

### Test Harness Scoping Architecture
Each evaluation executes in an automated, subprocess-isolated sandbox with a 5.0-second timeout guard under two evaluation architectures:
1. **AST-Scoped Clean Harness (Primary):** Strips any redundant target function redefinition from model-generated test code before execution, ensuring that unit test assertions execute strictly against benchmark reference implementations (C_ref) and faulty implementations (C_fault).
2. **Naive Concatenation Harness (Baseline):** Concatenates benchmark implementations directly with test code (`implementation + "\n\n" + test_code`), exposing how top-level function redefinitions silently shadow and overwrite benchmark functions at runtime.

### Evaluation Tiers
1. **Level 1 (Test Suite Usability M):** Executing generated tests against clean reference code C_ref. A test suite is usable (M = 1) iff it parses with valid Python syntax, executes without runtime exceptions, contains >= 1 non-trivial test assertions, and passes all assertions on C_ref.
2. **Level 2 (Defect Detection Y):** Executing usable suites against faulty code C_fault. A defect is detected (Y = 1) iff M = 1 and the suite fails specifically with an AssertionError on buggy logic:
   `Y = 1[M = 1 and Status(C_fault) == FAILED_ASSERTION]`

---

## 2. Directory Structure

```text
.
+-- paper.tex                               # Complete Springer Nature LaTeX manuscript
+-- paper.pdf                               # Compiled academic paper (0 overfull hboxes)
+-- references.bib                          # BibTeX bibliography with verified citations
+-- README.md                               # Replication package documentation
+-- data/
|   +-- verified_tasks_manifest.json        # 15 curated logic defect tasks manifest
|   +-- problems/                           # Detailed task problem specifications
+-- results_2048/                           # PRIMARY DATASET (2048-token unconstrained budget)
|   +-- experiment_records.csv              # 900-row tabular dataset (Clean Harness)
|   +-- experiment_records.jsonl            # 900-line valid JSONLines dataset (Clean Harness)
|   +-- experiment_records_naive_concat.jsonl # 900-line dataset under Naive Concatenation
|   +-- analysis_summary.json               # Full statistical summary artifact (Dual Harness)
+-- results/                                # SENSITIVITY PILOT DATASET (512-token constrained)
|   +-- experiment_records.csv              # Complete 900-row tabular pilot dataset
|   +-- experiment_records.jsonl            # Complete 900-line valid JSONLines pilot dataset
+-- scripts/
|   +-- harvest_tasks.py                    # Benchmark verification and ground-truth validation
|   +-- run_experiment_2048.py              # 2048-token live experiment runner (randomized order)
|   +-- analyze_results_2048.py             # Statistical analysis engine
|   +-- generate_dual_analysis_summary.py   # Comprehensive dual-harness statistical generator
|   +-- reevaluate_clean_harness.py         # AST scoping re-evaluation engine
|   +-- build_cleaned_records.py            # Primary dataset builder
+-- src/
    +-- sandbox.py                          # Subprocess-isolated evaluation environment
    +-- stats_utils.py                      # Pure-Python statistical engine
    +-- runner.py                           # Groq API client pool with multi-key rotation
    +-- schema.py                           # Data models and execution schemas
```

---

## 3. Key Empirical Findings (Primary 2048-Token Study)

### Descriptive Performance (N = 225 per condition, 900 total)

| Arm | Condition | Model | Redef. Fn | Clean Usable M | Clean Detect Y | Naive Detect Y | Pilot Y (512) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **T0** | `Spec-GPT` | `gpt-oss-120b` | 8/225 | 73.33% (165) | **72.89%** (164) | 69.78% (157) | 53.78% (121) |
| **T1** | `Code-GPT` | `gpt-oss-120b` | 88/225 | 85.78% (193) | **85.78%** (193) | 52.89% (119) | 21.78% (49) |
| **T2** | `Code-Qwen` | `qwen3.8-27b` | 155/225 | 72.89% (164) | **70.67%** (159) | 16.00% (36) | 14.22% (32) |
| **T3** | `Spec-Qwen` | `qwen3.8-27b` | 159/225 | 73.33% (165) | **71.11%** (160) | 15.11% (34) | 13.33% (30) |

### Task-Level Inferential Tests under Clean Scoped Harness (N = 15 clusters, df = 14)

1. **Code Exposure within GPT (T0 -> T1):**
   - Mean Delta: Delta = +12.89 pp (72.89% to 85.78%, SE = 6.81 pp).
   - Paired t-test: t(14) = 1.8935, p = 0.0791 (95% CI: [-1.71, +27.49] pp).
   - Holm-adjusted p = 0.2374. Exact permutation p = 0.0957. Wilcoxon W = 10.0 (p = 0.080).
   - 8 tasks improved with code, 5 were tied (3 at ceiling 15/15, TASK_009 at 7/15, TASK_012 at 9/15), and 2 decreased.
2. **Model Substitution Contrast Under Code Exposure (T1 -> T2):**
   - Mean Delta: Delta = -15.11 pp (85.78% to 70.67%, SE = 5.84 pp).
   - Paired t-test: t(14) = -2.5893, uncorrected p = 0.0214, Holm-adjusted p = 0.0857 (95% CI: [-27.63, -2.59] pp).
3. **Model Parity under Specification Only (T0 -> T3):**
   - Mean Delta: Delta = -1.78 pp (72.89% to 71.11%, SE = 5.61 pp, uncorrected p = 0.7562, Holm-adjusted p = 1.0000).
4. **Code Exposure within Qwen (T3 -> T2):**
   - Mean Delta: Delta = -0.44 pp (71.11% to 70.67%, SE = 5.18 pp, uncorrected p = 0.9329, Holm-adjusted p = 1.0000).

### Clustered Logistic Regression Estimates (AST-Scoped Clean Harness, df = 14)

```text
logit(P(Detect)) = Beta_0 + Beta_exp * Exposure + Beta_model * Model + Beta_inter * (Exposure * Model)
```

| Parameter | Coeff. (Beta) | Cluster SE_CR | t(14) | p-value (t_14) | Odds Ratio [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Intercept (Beta_0)** | +0.9890 | 0.3513 | +2.815 | **0.0138** | 2.6885 [1.2655, 5.7118] |
| **Code Exposure within GPT (Beta_exp)** | +0.8080 | 0.4080 | +1.980 | 0.0677 | 2.2433 [0.9351, 5.3817] |
| **Model: Qwen vs GPT at Spec (Beta_model)** | -0.0882 | 0.2822 | -0.313 | 0.7592 | 0.9156 [0.5000, 1.6766] |
| **Interaction: Ratio of Exposure ORs (Beta_inter)** | -0.8295 | 0.4912 | -1.689 | 0.1134 | 0.4363 [0.1521, 1.2514] |

### Descriptive Usability Funnel on Reference Code C_ref

| Condition & Harness | Usable (M=1) | Ref-Fail Assert | Runtime Crash | Syntax Error | Zero Asserts | Cond. Detect P(Y\|M) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean T0 (Spec-GPT)** | 73.33% (165) | 24.00% (54) | 2.22% (5) | 0.44% (1) | 0.00% (0) | 99.39% (164/165) |
| **Clean T1 (Code-GPT)** | 85.78% (193) | 11.56% (26) | 1.78% (4) | 0.44% (1) | 0.44% (1) | 100.00% (193/193) |
| **Clean T2 (Code-Qwen)** | 72.89% (164) | 13.78% (31) | 12.44% (28) | 0.89% (2) | 0.00% (0) | 96.95% (159/164) |
| **Clean T3 (Spec-Qwen)** | 73.33% (165) | 13.33% (30) | 8.44% (19) | 4.89% (11) | 0.00% (0) | 96.97% (160/165) |
| *Naive T1 (Code-GPT)* | 52.89% (119) | 44.44% (100) | 1.78% (4) | 0.44% (1) | 0.44% (1) | 100.00% (119/119) |
| *Naive T3 (Spec-Qwen)* | 69.78% (157) | 16.44% (37) | 8.89% (20) | 4.89% (11) | 0.00% (0) | 21.66% (34/157) |

---

## 4. Requirements & Dependencies

- **Python:** 3.10 or higher.
- **Dependencies:** **Python standard library only.** All statistical procedures are implemented directly in pure Python with zero third-party scientific dependencies (`numpy`, `scipy`, `pandas`, or `statsmodels` are NOT required).
- **LaTeX Environment (for paper compilation):** TeX Live or MiKTeX with standard packages (`article`, `geometry`, `amsmath`, `amssymb`, `booktabs`, `cite`, `microtype`, `graphicx`, `tabularx`, `hyperref`).

---

## 5. Reproduction Guide

### Step 1: Verify Task Benchmark Suite

```bash
python scripts/harvest_tasks.py
```

### Step 2: Reproduce Primary Statistical Analyses

```bash
python scripts/analyze_results_2048.py
```

### Step 3: Generate Full Dual-Harness Analysis Artifact

```bash
python scripts/generate_dual_analysis_summary.py
```

### Step 4: Compile Research Paper PDF

```bash
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

---

## 6. Cryptographic Data Verification (SHA-256)

To guarantee exact bitwise integrity of all empirical datasets and artifacts:

| File | SHA-256 Checksum |
| :--- | :--- |
| `results_2048/experiment_records.csv` | `a15ca20a13fdf19039516f761973fc862ebd1edbcd2c1f3c6940492a3b895649` |
| `results_2048/experiment_records.jsonl` | `1ec91182fb4b7aa6e937fbf0d9b638a833232cfa3266f5bfd78e75cca042bb1b` |
| `results_2048/experiment_records_naive_concat.jsonl` | `af95953394fa37c6f4b702a0d171406c353cbef48e132e58347515dc71f9e244` |
| `results_2048/analysis_summary.json` | `973598fcd11cf563b7fc35b323781e0461ac802e25f2a5ead98a176514dadba6` |
| `data/verified_tasks_manifest.json` | `ac1cf235c9c7921c630c57f4eccbc83fa67fee55274c2d18991c4d3e0b55828d` |
| `results/experiment_records.csv` | `9c9518aadae36ef11c0522df9ceb98120d7a0efdd5075fa021d05560ae6c016e` |
| `results/experiment_records.jsonl` | `934bde43828ddc9e603131c49a692423cea9ec01e5e56a67365c17fd7e5c95c9` |

To verify the integrity of the datasets locally, execute:

```bash
python -c "import hashlib; print('Clean JSONL:', hashlib.sha256(open('results_2048/experiment_records.jsonl', 'rb').read()).hexdigest()); print('Clean CSV:  ', hashlib.sha256(open('results_2048/experiment_records.csv', 'rb').read()).hexdigest())"
```
