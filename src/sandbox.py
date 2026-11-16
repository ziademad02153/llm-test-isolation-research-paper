"""
Execution sandbox with subprocess isolation, timeout guards, and assertion instrumentation.
"""

import sys
import os
import subprocess
import tempfile
import time
import ast
import re
from pathlib import Path
from typing import Tuple, Dict, Any, List


class SandboxStatus:
    PASSED = "PASSED"
    FAILED_ASSERTION = "FAILED_ASSERTION"
    RUNTIME_EXCEPTION = "RUNTIME_EXCEPTION"
    TIMEOUT = "TIMEOUT"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    SANDBOX_ERROR = "SANDBOX_ERROR"


def is_valid_syntax(code_str: str) -> Tuple[bool, str]:
    try:
        ast.parse(code_str)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"ParseError: {e}"


def execute_code_with_test(
    implementation_code: str,
    test_code: str,
    timeout_seconds: float = 5.0
) -> Tuple[str, str, float]:
    """
    Executes implementation + test in an isolated subprocess.
    Returns: (status, output, elapsed_ms)
    """
    # 1. Syntax check
    full_script = f"{implementation_code}\n\n{test_code}\n"
    valid, err_msg = is_valid_syntax(full_script)
    if not valid:
        return SandboxStatus.SYNTAX_ERROR, err_msg, 0.0

    # 2. Hardening wrapper
    harness = f"""# -*- coding: utf-8 -*-
import sys
import os

# Disable network access hooks
try:
    import socket
    socket.socket = None
    socket.create_connection = None
    socket.getaddrinfo = None
except Exception:
    pass

{implementation_code}

{test_code}
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(harness)
        tmp_path = tmp.name

    start_time = time.perf_counter()
    try:
        res = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            encoding="utf-8",
            errors="replace"
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if res.returncode == 0:
            return SandboxStatus.PASSED, res.stdout, elapsed_ms
        else:
            stderr = res.stderr
            if "AssertionError" in stderr:
                return SandboxStatus.FAILED_ASSERTION, stderr, elapsed_ms
            elif "SyntaxError" in stderr:
                return SandboxStatus.SYNTAX_ERROR, stderr, elapsed_ms
            else:
                return SandboxStatus.RUNTIME_EXCEPTION, stderr, elapsed_ms
    except subprocess.TimeoutExpired:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return SandboxStatus.TIMEOUT, f"Execution timed out after {timeout_seconds}s", elapsed_ms
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return SandboxStatus.SANDBOX_ERROR, str(e), elapsed_ms
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

