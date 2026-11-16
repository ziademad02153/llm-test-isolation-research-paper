"""
LLM Client pool supporting multi-key rotation on Groq, prompt building, and response parsing.
"""

import os
import sys
import json
import time
import re
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

project_root = Path(__file__).resolve().parent.parent

# Load .env if present
env_file = project_root / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()


class GroqPoolClient:
    def __init__(self, keys: Optional[List[str]] = None):
        if not keys:
            raw = os.environ.get("GROQ_KEYS", "")
            self.keys = [k.strip() for k in raw.split(",") if k.strip()]
        else:
            self.keys = keys

        if not self.keys:
            raise ValueError("No Groq keys found in environment or arguments.")

        self.current_idx = 0
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def _get_key(self) -> str:
        k = self.keys[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.keys)
        return k

    def generate(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        retries: int = 10
    ) -> Tuple[str, Dict[str, Any], str, str]:
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }).encode("utf-8")

        last_error = None
        for attempt in range(retries):
            api_key = self._get_key()
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Empirical-Study/1.0"
            }
            req = urllib.request.Request(self.endpoint, data=payload, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=35) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    choice = data["choices"][0]
                    msg = choice["message"]
                    content = msg.get("content") or msg.get("reasoning") or ""
                    finish_reason = choice.get("finish_reason") or "unknown"
                    usage = data.get("usage", {})
                    raw_completion = content
                    return content, usage, finish_reason, raw_completion
            except urllib.error.HTTPError as e:
                err_text = e.read().decode("utf-8", errors="replace")
                last_error = f"HTTP {e.code}: {err_text}"
                if e.code == 429:
                    match = re.search(r"Please try again in (\d+\.?\d*)s", err_text)
                    wait_time = float(match.group(1)) + 2.0 if match else 15.0
                    print(f"  [RateLimit 429] Backing off for {wait_time:.1f}s before retry (attempt {attempt+1}/{retries})...")
                    time.sleep(wait_time)
                else:
                    time.sleep(2.0)
            except Exception as e:
                last_error = str(e)
                print(f"  [Transient Network/DNS Error: {e}] Retrying in 4.0s (attempt {attempt+1}/{retries})...")
                time.sleep(4.0)

        raise RuntimeError(f"Failed after {retries} retries on Groq. Last error: {last_error}")


def extract_python_code(text: str) -> Tuple[str, bool, bool]:
    """
    Extracts python test code without silently discarding content.
    Returns: (cleaned_code, is_empty_suite, has_syntax_error)
    """
    import ast

    cleaned = text.strip()
    internal_blocks = re.findall(r"```(?:python)?(.*?)```", text, re.DOTALL)
    if internal_blocks:
        cleaned = max(internal_blocks, key=len).strip()
    else:
        cleaned = re.sub(r"^\s*```(?:python)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()

    if not cleaned:
        return "", True, False

    # Check AST parseability
    try:
        tree = ast.parse(cleaned)
        # Auto-invoke test functions if defined but not called
        test_funcs = [
            n.name for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
        ]
        invoked_calls = set()
        for n in tree.body:
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
                if isinstance(n.value.func, ast.Name):
                    invoked_calls.add(n.value.func.id)

        to_call = [f"{fn}()" for fn in test_funcs if fn not in invoked_calls]
        if to_call:
            cleaned += "\n\n" + "\n".join(to_call) + "\n"
        return cleaned, False, False
    except SyntaxError:
        # Check if only the very last incomplete line was truncated
        lines = cleaned.splitlines()
        if len(lines) > 1:
            test_candidate = "\n".join(lines[:-1]).strip()
            try:
                tree = ast.parse(test_candidate)
                test_funcs = [
                    n.name for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
                ]
                invoked_calls = set()
                for n in tree.body:
                    if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
                        if isinstance(n.value.func, ast.Name):
                            invoked_calls.add(n.value.func.id)

                to_call = [f"{fn}()" for fn in test_funcs if fn not in invoked_calls]
                if to_call:
                    test_candidate += "\n\n" + "\n".join(to_call) + "\n"
                return test_candidate, False, False
            except SyntaxError:
                pass

        # Preserve the code as-is; do NOT silently empty it
        return cleaned, False, True


def count_assert_statements(code: str) -> int:
    """Counts assertions using AST walk with regex fallback for syntax-errored code."""
    import ast
    if not code.strip():
        return 0
    try:
        tree = ast.parse(code)
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                count += 1
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr.startswith("assert"):
                    count += 1
        return count
    except SyntaxError:
        matches = re.findall(r"(?:^\s*assert\b|\.assert[A-Za-z0-9_]+\()", code, re.MULTILINE)
        return len(matches)


PROMPT_SPEC_ONLY = """You are an expert software testing engineer.
Write a concise suite of unit tests for the following Python function specification.

Function Signature:
{signature}

Specification:
\"\"\"
{docstring}
\"\"\"

Requirements:
1. Write 3 to 6 concise, targeted test assertions or test functions checking typical behavior and boundary conditions.
2. Directly invoke the function under test and assert the expected outputs.
3. Use plain Python `assert` statements (e.g. `assert function_name(arg) == expected`).
4. Output ONLY clean, executable Python code wrapped in a ```python ... ``` block. Do not include markdown commentary.
"""

PROMPT_CODE_FIRST = """You are an expert software testing engineer.
You are given a Python function specification AND its current implementation.
Write a concise suite of unit tests to verify the correctness of the function.

Function Signature:
{signature}

Specification:
\"\"\"
{docstring}
\"\"\"

Current Implementation:
```python
{implementation}
```

Requirements:
1. Write 3 to 6 concise, targeted test assertions or test functions checking typical behavior and boundary conditions.
2. Directly invoke the function under test and assert the expected outputs based strictly on the specification.
3. Use plain Python `assert` statements (e.g. `assert function_name(arg) == expected`).
4. Output ONLY clean, executable Python code wrapped in a ```python ... ``` block. Do not include markdown commentary.
"""
