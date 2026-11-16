"""
Task Harvester and Verification Engine.
Curates and verifies 15 diverse, net-new software engineering problem tasks
under strict acceptance criteria:
1. Syntactic viability (AST verified).
2. Pure Python standalone implementation (standard library only).
3. Golden test suite passes 100% on reference implementation.
4. Faulty implementation executes without syntax errors and fails at least one golden assertion.
"""

import os
import sys
import json
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.schema import ProblemSpec
from src.sandbox import execute_code_with_test, SandboxStatus, is_valid_syntax

TASKS_DEF = [
    {
        "problem_id": "TASK_001",
        "function_name": "parse_semver",
        "category": "Parsing/String",
        "difficulty": "Medium",
        "signature": "def parse_semver(version_str: str) -> Optional[Dict[str, Any]]:",
        "docstring": """Parse a Semantic Versioning 2.0.0 string into components.
Returns a dict with keys: 'major' (int), 'minor' (int), 'patch' (int), 'prerelease' (Optional[str]), 'build' (Optional[str]).
Returns None if the string is not a valid semver format (must have major.minor.patch as non-negative ints without leading zeroes).
Example: '1.2.3-alpha.1+build.001' -> {'major': 1, 'minor': 2, 'patch': 3, 'prerelease': 'alpha.1', 'build': 'build.001'}""",
        "ref_implementation": """from typing import Optional, Dict, Any
import re

def parse_semver(version_str: str) -> Optional[Dict[str, Any]]:
    if not isinstance(version_str, str) or not version_str.strip():
        return None
    pattern = r'^(0|[1-9]\\d*)\\.(0|[1-9]\\d*)\\.(0|[1-9]\\d*)(?:-([0-9A-Za-z.-]+))?(?:\\+([0-9A-Za-z.-]+))?$'
    match = re.match(pattern, version_str.strip())
    if not match:
        return None
    major, minor, patch, prerelease, build = match.groups()
    return {
        'major': int(major),
        'minor': int(minor),
        'patch': int(patch),
        'prerelease': prerelease if prerelease is not None else None,
        'build': build if build is not None else None
    }
""",
        "golden_test": """def test_parse_semver():
    res1 = parse_semver('1.2.3')
    assert res1 == {'major': 1, 'minor': 2, 'patch': 3, 'prerelease': None, 'build': None}
    res2 = parse_semver('0.10.4-rc.1+20261112')
    assert res2 == {'major': 0, 'minor': 10, 'patch': 4, 'prerelease': 'rc.1', 'build': '20261112'}
    assert parse_semver('01.2.3') is None
    assert parse_semver('1.2') is None
    assert parse_semver('v1.2.3') is None
    assert parse_semver('') is None

test_parse_semver()
""",
        "faulty_implementation": """from typing import Optional, Dict, Any

def parse_semver(version_str: str) -> Optional[Dict[str, Any]]:
    # Natural defect: Naive split on dots without validating leading zeroes or prerelease/build markers
    if not isinstance(version_str, str) or not version_str.strip():
        return None
    parts = version_str.strip().split('.')
    if len(parts) != 3:
        return None
    try:
        return {
            'major': int(parts[0]),
            'minor': int(parts[1]),
            'patch': int(parts[2]),
            'prerelease': None,
            'build': None
        }
    except ValueError:
        return None
""",
        "fault_type": "Boundary/Regex - Drops prerelease and build tags, accepts invalid leading zeroes",
        "fault_generating_model": "Natural_Coder_Defect"
    },
    {
        "problem_id": "TASK_002",
        "function_name": "merge_overlapping_intervals",
        "category": "Data Structures/Algorithms",
        "difficulty": "Medium",
        "signature": "def merge_overlapping_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:",
        "docstring": """Merge all overlapping or contiguous closed intervals.
Args:
    intervals: A list of tuples (start, end) where start <= end.
Returns:
    A new list of merged intervals sorted by start time.
Example: [(1, 3), (2, 6), (8, 10), (15, 18)] -> [(1, 6), (8, 10), (15, 18)]
Contiguous intervals like (1, 4) and (4, 7) should be merged into (1, 7).""",
        "ref_implementation": """from typing import List, Tuple

def merge_overlapping_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    for current in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        curr_start, curr_end = current
        if curr_start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged.append(current)
    return merged
""",
        "golden_test": """def test_merge_intervals():
    assert merge_overlapping_intervals([]) == []
    assert merge_overlapping_intervals([(1, 3), (2, 6), (8, 10), (15, 18)]) == [(1, 6), (8, 10), (15, 18)]
    assert merge_overlapping_intervals([(1, 4), (4, 5)]) == [(1, 5)]
    assert merge_overlapping_intervals([(1, 10), (2, 5), (6, 8)]) == [(1, 10)]
    assert merge_overlapping_intervals([(5, 7), (1, 3)]) == [(1, 3), (5, 7)]

test_merge_intervals()
""",
        "faulty_implementation": """from typing import List, Tuple

def merge_overlapping_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    # Natural defect: Strict inequality (< instead of <=) fails on contiguous intervals like (1, 4) and (4, 5)
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    for current in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        curr_start, curr_end = current
        if curr_start < prev_end:
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged.append(current)
    return merged
""",
        "fault_type": "Off-by-One - Strict inequality on boundary touch point",
        "fault_generating_model": "Natural_Coder_Defect"
    },
    {
        "problem_id": "TASK_003",
        "function_name": "calculate_exponential_backoff",
        "category": "Math/Networking",
        "difficulty": "Easy",
        "signature": "def calculate_exponential_backoff(attempt: int, base_delay: float, max_delay: float, factor: float = 2.0) -> float:",
        "docstring": """Calculate the deterministic exponential backoff delay for a given retry attempt.
Args:
    attempt: 0-indexed attempt count (0, 1, 2, ...). If attempt < 0, treated as 0.
    base_delay: Initial delay in seconds (must be >= 0.0).
    max_delay: Maximum capped delay in seconds (must be >= base_delay).
    factor: Multiplier per attempt (default 2.0).
Formula: min(max_delay, base_delay * (factor ** attempt)).
Returns:
    The delay clamped between base_delay and max_delay. If base_delay <= 0, returns 0.0.""",
        "ref_implementation": """def calculate_exponential_backoff(attempt: int, base_delay: float, max_delay: float, factor: float = 2.0) -> float:
    if base_delay <= 0.0:
        return 0.0
    att = max(0, attempt)
    delay = base_delay * (factor ** att)
    return max(base_delay, min(max_delay, delay))
""",
        "golden_test": """def test_backoff():
    assert calculate_exponential_backoff(0, 1.0, 10.0) == 1.0
    assert calculate_exponential_backoff(1, 1.0, 10.0) == 2.0
    assert calculate_exponential_backoff(2, 1.0, 10.0) == 4.0
    assert calculate_exponential_backoff(3, 1.0, 10.0) == 8.0
    assert calculate_exponential_backoff(4, 1.0, 10.0) == 10.0
    assert calculate_exponential_backoff(-1, 1.0, 10.0) == 1.0
    assert calculate_exponential_backoff(5, 0.0, 10.0) == 0.0

test_backoff()
""",
        "faulty_implementation": """def calculate_exponential_backoff(attempt: int, base_delay: float, max_delay: float, factor: float = 2.0) -> float:
    # Natural defect: 1-indexed exponent assumption (factor ** (attempt + 1)), causing premature spike
    if base_delay <= 0.0:
        return 0.0
    att = max(0, attempt)
    delay = base_delay * (factor ** (att + 1))
    return min(max_delay, delay)
""",
        "fault_type": "Index/Exponent shift - Attempt 0 yields base_delay * 2 instead of base_delay",
        "fault_generating_model": "Natural_Coder_Defect"
    },
    {
        "problem_id": "TASK_004",
        "function_name": "flatten_nested_dict",
        "category": "Data Structures",
        "difficulty": "Medium",
        "signature": "def flatten_nested_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:",
        "docstring": """Flatten a hierarchically nested dictionary into a single-level dictionary with compound keys.
Args:
    d: The input dictionary.
    sep: Separator string for joining nested keys (default '.').
Returns:
    A new flattened dictionary where nested keys are joined by sep.
    Empty dictionaries at leaf nodes are preserved as empty dicts.
Example: {'a': {'b': 1, 'c': {'d': 2}}, 'e': 3} -> {'a.b': 1, 'a.c.d': 2, 'e': 3}""",
        "ref_implementation": """from typing import Dict, Any

def flatten_nested_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    items = {}
    def _flatten(current: Any, parent_key: str):
        if isinstance(current, dict) and current:
            for k, v in current.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
                _flatten(v, new_key)
        else:
            if parent_key:
                items[parent_key] = current
    _flatten(d, '')
    return items
""",
        "golden_test": """def test_flatten():
    assert flatten_nested_dict({}) == {}
    assert flatten_nested_dict({'a': 1, 'b': 2}) == {'a': 1, 'b': 2}
    nested = {'a': {'b': 1, 'c': {'d': 2}}, 'e': 3}
    assert flatten_nested_dict(nested) == {'a.b': 1, 'a.c.d': 2, 'e': 3}
    custom_sep = {'db': {'host': 'localhost', 'port': 5432}}
    assert flatten_nested_dict(custom_sep, sep='__') == {'db__host': 'localhost', 'db__port': 5432}
    empty_leaf = {'a': {}}
    assert flatten_nested_dict(empty_leaf) == {'a': {}}

test_flatten()
""",
        "faulty_implementation": """from typing import Dict, Any

def flatten_nested_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    # Natural defect: Fails to preserve empty leaf dicts, discarding them completely
    items = {}
    def _flatten(current: Any, parent_key: str):
        if isinstance(current, dict):
            for k, v in current.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
                _flatten(v, new_key)
        else:
            items[parent_key] = current
    _flatten(d, '')
    return items
""",
        "fault_type": "Missing Base Case - Silently drops empty dict leaf values",
        "fault_generating_model": "Natural_Coder_Defect"
    },
    {
        "problem_id": "TASK_005",
        "function_name": "validate_balanced_parentheses",
        "category": "Parsing/Stack",
        "difficulty": "Easy",
        "signature": "def validate_balanced_parentheses(s: str) -> bool:",
        "docstring": """Validate whether brackets in a string are balanced and properly nested.
Supported bracket pairs are: '()', '[]', '{}'.
Non-bracket characters are ignored.
Returns True if all brackets are properly paired and closed in correct order, False otherwise.
Example: 'a(b[c]d)e' -> True, '([)]' -> False, '{{}' -> False.""",
        "ref_implementation": """def validate_balanced_parentheses(s: str) -> bool:
    if not isinstance(s, str):
        return False
    stack = []
    mapping = {')': '(', ']': '[', '}': '{'}
    opening = set(mapping.values())
    for char in s:
        if char in opening:
            stack.append(char)
        elif char in mapping:
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()
    return len(stack) == 0
""",
        "golden_test": """def test_brackets():
    assert validate_balanced_parentheses('') is True
    assert validate_balanced_parentheses('()[]{}') is True
    assert validate_balanced_parentheses('([{}])') is True
    assert validate_balanced_parentheses('text (with [nested {brackets}]!)') is True
    assert validate_balanced_parentheses('(]') is False
    assert validate_balanced_parentheses('([)]') is False
    assert validate_balanced_parentheses('(((') is False
    assert validate_balanced_parentheses(')') is False

test_brackets()
""",
        "faulty_implementation": """def validate_balanced_parentheses(s: str) -> bool:
    # Natural defect: Counts bracket frequencies instead of enforcing stack order (fails on '([)]')
    if not isinstance(s, str):
        return False
    return (s.count('(') == s.count(')') and
            s.count('[') == s.count(']') and
            s.count('{') == s.count('}'))
""",
        "fault_type": "Structural/Stack Fallacy - Pure counting ignores interleaved nesting order",
        "fault_generating_model": "Natural_Coder_Defect"
    },
    {
        "problem_id": "TASK_006",
        "function_name": "canonicalize_query_string",
        "category": "Parsing/String",
        "difficulty": "Medium",
        "signature": "def canonicalize_query_string(url_or_query: str, drop_keys: Optional[List[str]] = None) -> str:",
        "docstring": """Extract and canonicalize the URL query string parameters.
1. Extracts query string from URL (if '?' is present) or takes raw query string.
2. Drops any query parameters whose key is in drop_keys (case-insensitive comparison).
3. Sorts parameter keys alphabetically. For keys with multiple values, preserves their original order.
4. Returns the reconstituted canonical query string (e.g. 'a=1&b=2'). Returns '' if no parameters remain.""",
        "ref_implementation": """from typing import Optional, List
from urllib.parse import parse_qs, urlencode, urlparse

def canonicalize_query_string(url_or_query: str, drop_keys: Optional[List[str]] = None) -> str:
    if not url_or_query:
        return ""
    if "?" in url_or_query:
        query_part = url_or_query.split("?", 1)[1]
    else:
        query_part = url_or_query
    
    if "#" in query_part:
        query_part = query_part.split("#", 1)[0]
        
    parsed = parse_qs(query_part, keep_blank_values=True)
    drop_set = {k.lower() for k in drop_keys} if drop_keys else set()
    
    filtered = {}
    for k in sorted(parsed.keys()):
        if k.lower() not in drop_set:
            filtered[k] = parsed[k]
            
    pairs = []
    for k in sorted(filtered.keys()):
        for v in filtered[k]:
            pairs.append((k, v))
            
    return urlencode(pairs)
""",
        "golden_test": """def test_canonical():
    assert canonicalize_query_string('https://api.test.com/v1?b=2&a=1') == 'a=1&b=2'
    assert canonicalize_query_string('utm_source=fb&q=python&utm_medium=cpc', drop_keys=['utm_source', 'utm_medium']) == 'q=python'
    assert canonicalize_query_string('UTM_SOURCE=fb&q=python', drop_keys=['utm_source']) == 'q=python'
    assert canonicalize_query_string('c=3&a=1&b=2') == 'a=1&b=2&c=3'
    assert canonicalize_query_string('empty_val=&x=1') == 'empty_val=&x=1'
    assert canonicalize_query_string('') == ''

test_canonical()
""",
        "faulty_implementation": """from typing import Optional, List
from urllib.parse import parse_qs, urlencode

def canonicalize_query_string(url_or_query: str, drop_keys: Optional[List[str]] = None) -> str:
    # Natural defect: Case-sensitive drop check (fails to drop 'UTM_SOURCE' when drop_keys=['utm_source'])
    if not url_or_query:
        return ""
    query_part = url_or_query.split("?", 1)[1] if "?" in url_or_query else url_or_query
    parsed = parse_qs(query_part, keep_blank_values=True)
    drop_set = set(drop_keys) if drop_keys else set()
    pairs = []
    for k in sorted(parsed.keys()):
        if k not in drop_set:
            for v in parsed[k]:
                pairs.append((k, v))
    return urlencode(pairs)
""",
        "fault_type": "Case-Sensitivity Oversight - Case-sensitive filtering on case-insensitive protocol parameters",
        "fault_generating_model": "Natural_Coder_Defect"
    },
    {
        "problem_id": "TASK_007",
        "function_name": "topological_sort_tasks",
        "category": "Algorithms/Graphs",
        "difficulty": "Hard",
        "signature": "def topological_sort_tasks(dependencies: Dict[str, List[str]]) -> Optional[List[str]]:",
        "docstring": """Compute a valid execution order for tasks given their prerequisite dependencies.
Args:
    dependencies: Dict mapping task_name -> list of prerequisite task_names that must run before it.
Returns:
    A list of all task names in valid execution order.
    If multiple valid orders exist, ties should be broken deterministically by alphabetical order.
    If a circular dependency cycle exists, returns None.
Example: {'deploy': ['build', 'test'], 'build': [], 'test': ['build']} -> ['build', 'test', 'deploy']""",
        "ref_implementation": """from typing import Dict, List, Optional
import heapq

def topological_sort_tasks(dependencies: Dict[str, List[str]]) -> Optional[List[str]]:
    all_nodes = set(dependencies.keys())
    for deps in dependencies.values():
        all_nodes.update(deps)
        
    adj = {node: [] for node in all_nodes}
    in_degree = {node: 0 for node in all_nodes}
    
    for task, prereqs in dependencies.items():
        for prereq in prereqs:
            adj[prereq].append(task)
            in_degree[task] += 1
            
    heap = [node for node, deg in in_degree.items() if deg == 0]
    heapq.heapify(heap)
    
    order = []
    while heap:
        curr = heapq.heappop(heap)
        order.append(curr)
        for neighbor in adj[curr]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                heapq.heappush(heap, neighbor)
                
    if len(order) != len(all_nodes):
        return None
    return order
""",
        "golden_test": """def test_topo():
    deps1 = {'deploy': ['build', 'test'], 'build': [], 'test': ['build']}
    assert topological_sort_tasks(deps1) == ['build', 'test', 'deploy']
    cycle = {'a': ['b'], 'b': ['c'], 'c': ['a']}
    assert topological_sort_tasks(cycle) is None
    independent = {'z': [], 'a': [], 'm': []}
    assert topological_sort_tasks(independent) == ['a', 'm', 'z']
    assert topological_sort_tasks({}) == []

test_topo()
""",
        "faulty_implementation": """from typing import Dict, List, Optional

def topological_sort_tasks(dependencies: Dict[str, List[str]]) -> Optional[List[str]]:
    # Natural defect: Inverts dependency direction (treats map as task -> downstream dependents)
    all_nodes = sorted(list(set(dependencies.keys()).union(*(dependencies.values())))) if dependencies else []
    in_degree = {n: 0 for n in all_nodes}
    adj = {n: [] for n in all_nodes}
    for task, prereqs in dependencies.items():
        for p in prereqs:
            adj[task].append(p)
            in_degree[p] += 1
    queue = [n for n in all_nodes if in_degree[n] == 0]
    res = []
    while queue:
        curr = queue.pop(0)
        res.append(curr)
        for nbr in adj[curr]:
            in_degree[nbr] -= 1
            if in_degree[nbr] == 0:
                queue.append(nbr)
    return res if len(res) == len(all_nodes) else None
""",
        "fault_type": "Graph Direction Inversion - Treats prerequisites as downstream dependents",
        "fault_generating_model": "Natural_Coder_Defect"
    },
    {
        "problem_id": "TASK_008",
        "function_name": "rate_limiter_sliding_window",
        "category": "State/Algorithms",
        "difficulty": "Medium",
        "signature": "def rate_limiter_sliding_window(history: List[float], new_timestamp: float, window_seconds: float, max_requests: int) -> Tuple[bool, List[float]]:",
        "docstring": """Simulate a sliding window rate limiter check for an incoming request.
Args:
    history: List of timestamps (in seconds) of previously admitted requests, assumed sorted.
    new_timestamp: Timestamp of the candidate incoming request.
    window_seconds: Duration of the sliding time window.
    max_requests: Maximum allowed requests strictly within (new_timestamp - window_seconds, new_timestamp].
Returns:
    Tuple of (is_allowed: bool, updated_history: List[float]).
    If allowed, updated_history includes new_timestamp and excludes expired timestamps outside the window.
    If rejected, updated_history contains all valid timestamps within the window without new_timestamp.""",
        "ref_implementation": """from typing import List, Tuple

def rate_limiter_sliding_window(history: List[float], new_timestamp: float, window_seconds: float, max_requests: int) -> Tuple[bool, List[float]]:
    cutoff = new_timestamp - window_seconds
    valid_history = [ts for ts in history if ts > cutoff]
    if len(valid_history) < max_requests:
        valid_history.append(new_timestamp)
        return True, valid_history
    else:
        return False, valid_history
""",
        "golden_test": """def test_rate_limiter():
    allowed, hist = rate_limiter_sliding_window([], 10.0, 60.0, 3)
    assert allowed is True and hist == [10.0]
    allowed, hist2 = rate_limiter_sliding_window([5.0, 8.0], 10.0, 60.0, 3)
    assert allowed is True and hist2 == [5.0, 8.0, 10.0]
    allowed, hist3 = rate_limiter_sliding_window([2.0, 5.0, 8.0], 10.0, 60.0, 3)
    assert allowed is False and hist3 == [2.0, 5.0, 8.0]
    # Expired events pruned
    allowed, hist4 = rate_limiter_sliding_window([1.0, 2.0, 100.0], 105.0, 10.0, 2)
    assert allowed is True and hist4 == [100.0, 105.0]

test_rate_limiter()
""",
        "faulty_implementation": """from typing import List, Tuple

def rate_limiter_sliding_window(history: List[float], new_timestamp: float, window_seconds: float, max_requests: int) -> Tuple[bool, List[float]]:
    # Natural defect: Uses <= max_requests instead of < max_requests, allowing (max_requests + 1) requests
    cutoff = new_timestamp - window_seconds
    valid_history = [ts for ts in history if ts > cutoff]
    if len(valid_history) <= max_requests:
        valid_history.append(new_timestamp)
        return True, valid_history
    return False, valid_history
""",
        "fault_type": "Capacity Off-by-One - Admits N+1 requests instead of capping strictly at N",
        "fault_generating_model": "Natural_Coder_Defect"
    },
    {
        "problem_id": "TASK_009",
        "function_name": "parse_csv_line_with_quotes",
        "category": "Parsing/String",
        "difficulty": "Hard",
        "signature": "def parse_csv_line_with_quotes(line: str, delimiter: str = ',') -> List[str]:",
        "docstring": """Parse a single CSV row supporting double-quote escaping and embedded delimiters.
Rules:
- Fields separated by delimiter.
- Fields enclosed in double quotes can contain the delimiter and escaped quotes (represented as '\"\"').
- Enclosing quotes are stripped from the resulting field.
- Returns list of field strings with whitespace preserved.
Example: '1,\"Smith, John\",\"\"\"Hero\"\"\"' -> ['1', 'Smith, John', '\"Hero\"']""",
        "ref_implementation": """from typing import List

def parse_csv_line_with_quotes(line: str, delimiter: str = ',') -> List[str]:
    fields = []
    current = []
    in_quotes = False
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c == '"':
            if in_quotes and i + 1 < n and line[i + 1] == '"':
                current.append('"')
                i += 1
            else:
                in_quotes = not in_quotes
        elif c == delimiter and not in_quotes:
            fields.append(''.join(current))
            current = []
        else:
            current.append(c)
        i += 1
    fields.append(''.join(current))
    return fields
""",
        "golden_test": '''def test_csv():
    assert parse_csv_line_with_quotes('a,b,c') == ['a', 'b', 'c']
    assert parse_csv_line_with_quotes('1,"Smith, John",active') == ['1', 'Smith, John', 'active']
    assert parse_csv_line_with_quotes('quote,"He said \\"\\"hello\\"\\"",end') == ['quote', 'He said "hello"', 'end']
    assert parse_csv_line_with_quotes('') == ['']
    assert parse_csv_line_with_quotes('one|two|three', delimiter='|') == ['one', 'two', 'three']

test_csv()
''',
        "faulty_implementation": """from typing import List

def parse_csv_line_with_quotes(line: str, delimiter: str = ',') -> List[str]:
    # Natural defect: Naive split on delimiter, completely failing to handle quoted commas
    return [f.strip('\"') for f in line.split(delimiter)]
""",
        "fault_type": "Delimiter Blindness - Fails to protect delimiters inside quoted strings",
        "fault_generating_model": "Natural_Coder_Defect"
    },
    {
        "problem_id": "TASK_010",
        "function_name": "lru_cache_simulator",
        "category": "Data Structures",
        "difficulty": "Medium",
        "signature": "def lru_cache_simulator(capacity: int, operations: List[Tuple[str, str, Optional[str]]]) -> List[Optional[str]]:",
        "docstring": """Simulate an LRU (Least Recently Used) cache with fixed capacity.
Args:
    capacity: Maximum key-value pairs stored (must be > 0).
    operations: List of tuples representing actions:
        - ('GET', key, None) -> returns stored value or None if not found. Updates access order.
        - ('PUT', key, value) -> inserts/updates key with value. Evicts least recently used if at capacity. Returns None.
Returns:
    List of results corresponding to each operation (None for PUT, value/None for GET).""",
        "ref_implementation": """from typing import List, Tuple, Optional
from collections import OrderedDict

def lru_cache_simulator(capacity: int, operations: List[Tuple[str, str, Optional[str]]]) -> List[Optional[str]]:
    cache = OrderedDict()
    results = []
    for op, key, val in operations:
        if op == 'GET':
            if key in cache:
                cache.move_to_end(key)
                results.append(cache[key])
            else:
                results.append(None)
        elif op == 'PUT':
            if key in cache:
                cache[key] = val
                cache.move_to_end(key)
            else:
                if len(cache) >= capacity:
                    cache.popitem(last=False)
                cache[key] = val
            results.append(None)
    return results
""",
        "golden_test": """def test_lru():
    ops = [
        ('PUT', 'a', '1'),
        ('PUT', 'b', '2'),
        ('GET', 'a', None),
        ('PUT', 'c', '3'), # 'b' evicted
        ('GET', 'b', None),
        ('GET', 'c', None),
        ('GET', 'a', None)
    ]
    assert lru_cache_simulator(2, ops) == [None, None, '1', None, None, '3', '1']

test_lru()
""",
        "faulty_implementation": """from typing import List, Tuple, Optional

def lru_cache_simulator(capacity: int, operations: List[Tuple[str, str, Optional[str]]]) -> List[Optional[str]]:
    # Natural defect: Regular dict without access refresh on GET, causing FIFO instead of LRU eviction
    cache = {}
    results = []
    for op, key, val in operations:
        if op == 'GET':
            results.append(cache.get(key, None))
        elif op == 'PUT':
            if key not in cache and len(cache) >= capacity:
                first_key = next(iter(cache))
                del cache[first_key]
            cache[key] = val
            results.append(None)
    return results
""",
        "fault_type": "Cache Policy Drift - Implements FIFO instead of LRU (GET does not update recency)",
        "fault_generating_model": "Natural_Coder_Defect"
    },
    {
        "problem_id": "TASK_011",
        "function_name": "roman_to_integer",
        "category": "Algorithms/Math",
        "difficulty": "Easy",
        "signature": "def roman_to_integer(s: str) -> Optional[int]:",
        "docstring": """Convert a valid Roman numeral string to its integer representation.
Valid Roman symbols: I=1, V=5, X=10, L=50, C=100, D=500, M=1000.
Subtractive notation: IV=4, IX=9, XL=40, XC=90, CD=400, CM=900.
Returns integer value, or None if string contains invalid characters or is empty.""",
        "ref_implementation": """from typing import Optional

def roman_to_integer(s: str) -> Optional[int]:
    if not isinstance(s, str) or not s:
        return None
    val_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    prev = 0
    for char in reversed(s.upper()):
        if char not in val_map:
            return None
        curr = val_map[char]
        if curr < prev:
            total -= curr
        else:
            total += curr
        prev = curr
    return total
""",
        "golden_test": """def test_roman():
    assert roman_to_integer('III') == 3
    assert roman_to_integer('IV') == 4
    assert roman_to_integer('IX') == 9
    assert roman_to_integer('LVIII') == 58
    assert roman_to_integer('MCMXCIV') == 1994
    assert roman_to_integer('') is None
    assert roman_to_integer('XYZ') is None

test_roman()
""",
        "faulty_implementation": """from typing import Optional

def roman_to_integer(s: str) -> Optional[int]:
    # Natural defect: Naive forward summation without handling subtractive pairs (e.g. IV becomes 1+5=6)
    if not isinstance(s, str) or not s:
        return None
    val_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    for c in s.upper():
        if c not in val_map:
            return None
        total += val_map[c]
    return total
""",
        "fault_type": "Subtractive Ignorance - Forward sum treats IV as 6 instead of 4",
        "fault_generating_model": "Natural_Coder_Defect"
    },
    {
        "problem_id": "TASK_012",
        "function_name": "find_available_slots",
        "category": "Algorithms/Scheduling",
        "difficulty": "Medium",
        "signature": "def find_available_slots(busy_intervals: List[Tuple[int, int]], day_start: int, day_end: int, min_duration: int) -> List[Tuple[int, int]]:",
        "docstring": """Find all free continuous time slots during a workday that are at least min_duration long.
Args:
    busy_intervals: List of (start, end) busy times during the day.
    day_start: Beginning of workday (e.g. 540 for 9:00 AM in minutes).
    day_end: End of workday (e.g. 1020 for 5:00 PM).
    min_duration: Minimum required free slot duration in minutes.
Returns:
    List of (start, end) free intervals sorted chronologically.""",
        "ref_implementation": """from typing import List, Tuple

def find_available_slots(busy_intervals: List[Tuple[int, int]], day_start: int, day_end: int, min_duration: int) -> List[Tuple[int, int]]:
    if day_start >= day_end or min_duration <= 0:
        return []
    # Clip and filter busy slots
    clipped = []
    for s, e in busy_intervals:
        cs = max(day_start, s)
        ce = min(day_end, e)
        if cs < ce:
            clipped.append((cs, ce))
    clipped.sort(key=lambda x: x[0])
    
    # Merge overlapping busy slots
    merged = []
    for interval in clipped:
        if not merged or interval[0] > merged[-1][1]:
            merged.append(interval)
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))
            
    free_slots = []
    current_time = day_start
    for bs, be in merged:
        if bs - current_time >= min_duration:
            free_slots.append((current_time, bs))
        current_time = max(current_time, be)
        
    if day_end - current_time >= min_duration:
        free_slots.append((current_time, day_end))
        
    return free_slots
""",
        "golden_test": """def test_slots():
    busy = [(600, 660), (720, 780)]
    assert find_available_slots(busy, 540, 1020, 30) == [(540, 600), (660, 720), (780, 1020)]
    # Merged overlapping busy
    busy2 = [(550, 600), (580, 650)]
    assert find_available_slots(busy2, 540, 700, 40) == [(650, 700)]
    # Entire day busy
    assert find_available_slots([(500, 1100)], 540, 1020, 15) == []

test_slots()
""",
        "faulty_implementation": """from typing import List, Tuple

def find_available_slots(busy_intervals: List[Tuple[int, int]], day_start: int, day_end: int, min_duration: int) -> List[Tuple[int, int]]:
    # Natural defect: Fails to check trailing interval between last busy slot and day_end
    if day_start >= day_end or min_duration <= 0:
        return []
    sorted_busy = sorted(busy_intervals, key=lambda x: x[0])
    free_slots = []
    curr = day_start
    for s, e in sorted_busy:
        if s - curr >= min_duration:
            free_slots.append((curr, s))
        curr = max(curr, e)
    return free_slots
""",
        "fault_type": "Missing Trailing Boundary - Ignores remaining free time after final meeting",
        "fault_generating_model": "Natural_Coder_Defect"
    },
    {
        "problem_id": "TASK_013",
        "function_name": "spiral_matrix_traverse",
        "category": "Algorithms/Arrays",
        "difficulty": "Medium",
        "signature": "def spiral_matrix_traverse(matrix: List[List[int]]) -> List[int]:",
        "docstring": """Traverse an M x N 2D matrix in clockwise spiral order.
Order: top row left-to-right, right column top-to-bottom, bottom row right-to-left, left column bottom-to-top.
Returns:
    List of integer elements in spiral sequence. Returns [] for empty matrix.""",
        "ref_implementation": """from typing import List

def spiral_matrix_traverse(matrix: List[List[int]]) -> List[int]:
    if not matrix or not matrix[0]:
        return []
    res = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    
    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            res.append(matrix[top][c])
        top += 1
        
        for r in range(top, bottom + 1):
            res.append(matrix[r][right])
        right -= 1
        
        if top <= bottom:
            for c in range(right, left - 1, -1):
                res.append(matrix[bottom][c])
            bottom -= 1
            
        if left <= right:
            for r in range(bottom, top - 1, -1):
                res.append(matrix[r][left])
            left += 1
            
    return res
""",
        "golden_test": """def test_spiral():
    m1 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    assert spiral_matrix_traverse(m1) == [1, 2, 3, 6, 9, 8, 7, 4, 5]
    m2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert spiral_matrix_traverse(m2) == [1, 2, 3, 4, 8, 7, 6, 5]
    assert spiral_matrix_traverse([[1, 2, 3]]) == [1, 2, 3]
    assert spiral_matrix_traverse([]) == []

test_spiral()
""",
        "faulty_implementation": """from typing import List

def spiral_matrix_traverse(matrix: List[List[int]]) -> List[int]:
    # Natural defect: Omits inner boundary check (if top <= bottom and left <= right) before reverse sweeps
    if not matrix or not matrix[0]:
        return []
    res = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            res.append(matrix[top][c])
        top += 1
        for r in range(top, bottom + 1):
            res.append(matrix[r][right])
        right -= 1
        # Missing check causes duplicates on non-square matrices like 2x4
        for c in range(right, left - 1, -1):
            res.append(matrix[bottom][c])
        bottom -= 1
        for r in range(bottom, top - 1, -1):
            res.append(matrix[r][left])
        left += 1
    return res
""",
        "fault_type": "Loop Invariant Violation - Generates duplicate elements on rectangular matrices",
        "fault_generating_model": "Natural_Coder_Defect"
    },
    {
        "problem_id": "TASK_014",
        "function_name": "parse_markdown_table_to_dicts",
        "category": "Parsing/Text",
        "difficulty": "Medium",
        "signature": "def parse_markdown_table_to_dicts(md_table: str) -> List[Dict[str, str]]:",
        "docstring": """Parse a GitHub Flavored Markdown table into a list of row dictionaries.
Format:
- Line 1: Header names separated by '|'.
- Line 2: Delimiter line (e.g. '|---|---|') which must be skipped.
- Subsequent lines: Data rows.
Rules:
- Leading and trailing pipes on each row are optional.
- Whitespace inside cell values is stripped.
- Skips empty lines.
Returns list of dicts mapping header string -> cell string value.""",
        "ref_implementation": """from typing import List, Dict

def parse_markdown_table_to_dicts(md_table: str) -> List[Dict[str, str]]:
    if not md_table or not md_table.strip():
        return []
    lines = [line.strip() for line in md_table.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        return []
    
    def _split_row(row: str) -> List[str]:
        cleaned = row
        if cleaned.startswith('|'):
            cleaned = cleaned[1:]
        if cleaned.endswith('|'):
            cleaned = cleaned[:-1]
        return [c.strip() for c in cleaned.split('|')]
        
    headers = _split_row(lines[0])
    rows = []
    for line in lines[2:]:
        cells = _split_row(line)
        row_dict = {}
        for i, h in enumerate(headers):
            row_dict[h] = cells[i] if i < len(cells) else ""
        rows.append(row_dict)
    return rows
""",
        "golden_test": """def test_md_table():
    tbl = '''| Name | Age | Role |
| --- | --- | --- |
| Alice | 30 | Admin |
| Bob | 25 | User |'''
    expected = [
        {'Name': 'Alice', 'Age': '30', 'Role': 'Admin'},
        {'Name': 'Bob', 'Age': '25', 'Role': 'User'}
    ]
    assert parse_markdown_table_to_dicts(tbl) == expected
    assert parse_markdown_table_to_dicts('') == []

test_md_table()
""",
        "faulty_implementation": """from typing import List, Dict

def parse_markdown_table_to_dicts(md_table: str) -> List[Dict[str, str]]:
    # Natural defect: Starts parsing data from line 1 instead of skipping separator line 2
    if not md_table or not md_table.strip():
        return []
    lines = [l.strip() for l in md_table.splitlines() if l.strip()]
    if len(lines) < 2:
        return []
    headers = [h.strip() for h in lines[0].strip('|').split('|')]
    rows = []
    for line in lines[1:]: # BUG: Includes '| --- | --- |' as row
        cells = [c.strip() for c in line.strip('|').split('|')]
        rows.append({h: cells[i] if i < len(cells) else '' for i, h in enumerate(headers)})
    return rows
""",
        "fault_type": "Off-by-One - Includes markdown separator line as data row",
        "fault_generating_model": "Natural_Coder_Defect"
    },
    {
        "problem_id": "TASK_015",
        "function_name": "compute_levenshtein_distance",
        "category": "Algorithms/Dynamic Programming",
        "difficulty": "Medium",
        "signature": "def compute_levenshtein_distance(s1: str, s2: str) -> int:",
        "docstring": """Compute the minimum Levenshtein edit distance between two strings s1 and s2.
Allowed operations (each with cost 1): insertion, deletion, substitution.
Returns integer edit distance (0 if s1 == s2).""",
        "ref_implementation": """def compute_levenshtein_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],      # deletion
                                   dp[i][j - 1],      # insertion
                                   dp[i - 1][j - 1])  # substitution
    return dp[m][n]
""",
        "golden_test": """def test_levenshtein():
    assert compute_levenshtein_distance('', '') == 0
    assert compute_levenshtein_distance('kitten', 'sitting') == 3
    assert compute_levenshtein_distance('flaw', 'lawn') == 2
    assert compute_levenshtein_distance('same', 'same') == 0
    assert compute_levenshtein_distance('a', '') == 1

test_levenshtein()
""",
        "faulty_implementation": """def compute_levenshtein_distance(s1: str, s2: str) -> int:
    # Natural defect: Always adds substitution cost 1 even when characters match (ignores s1[i-1] == s2[j-1])
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]
""",
        "fault_type": "Recurrence Invariant Broken - Unconditionally increments substitution cost",
        "fault_generating_model": "Natural_Coder_Defect"
    }
]


def verify_and_save_all():
    print(f"Starting verification of {len(TASKS_DEF)} candidate tasks...")
    problems_dir = project_root / "data" / "problems"
    problems_dir.mkdir(parents=True, exist_ok=True)
    
    verified_tasks = []
    
    for t in TASKS_DEF:
        pid = t["problem_id"]
        fn = t["function_name"]
        
        # 1. Check syntax of ref
        ok, err = is_valid_syntax(t["ref_implementation"])
        assert ok, f"Ref syntax error in {pid}: {err}"
        
        # 2. Check syntax of faulty
        ok, err = is_valid_syntax(t["faulty_implementation"])
        assert ok, f"Faulty syntax error in {pid}: {err}"
        
        # 3. Check golden test syntax
        ok, err = is_valid_syntax(t["golden_test"])
        assert ok, f"Golden test syntax error in {pid}: {err}"
        
        # 4. Run ref + golden test -> MUST PASS
        status, out, ms = execute_code_with_test(t["ref_implementation"], t["golden_test"])
        if status != SandboxStatus.PASSED:
            print(f"FAIL on REF for {pid}: {status}\n{out}")
            sys.exit(1)
            
        # 5. Run faulty + golden test -> MUST FAIL ASSERTION
        status, out, ms = execute_code_with_test(t["faulty_implementation"], t["golden_test"])
        if status != SandboxStatus.FAILED_ASSERTION:
            print(f"FAIL on FAULTY check for {pid}: expected FAILED_ASSERTION, got {status}\n{out}")
            sys.exit(1)
            
        # Task is 100% verified!
        task_obj = ProblemSpec(
            problem_id=pid,
            function_name=fn,
            signature=t["signature"],
            docstring=t["docstring"],
            ref_implementation=t["ref_implementation"],
            golden_test=t["golden_test"],
            category=t["category"],
            difficulty=t["difficulty"],
            faulty_implementation=t["faulty_implementation"],
            fault_type=t["fault_type"],
            fault_generating_model=t["fault_generating_model"]
        )
        
        out_path = problems_dir / f"{pid}.json"
        out_path.write_text(json.dumps(task_obj.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        verified_tasks.append(task_obj.to_dict())
        print(f"  [PASS] {pid} ({fn}): Category='{t['category']}', Defect='{t['fault_type'][:40]}...'")

    # Save summary manifest
    manifest_path = project_root / "data" / "verified_tasks_manifest.json"
    manifest_path.write_text(json.dumps(verified_tasks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nALL {len(verified_tasks)} TASKS VERIFIED AND PERSISTED TO DISK!")


if __name__ == "__main__":
    verify_and_save_all()
