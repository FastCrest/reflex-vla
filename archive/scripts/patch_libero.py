#!/usr/bin/env python3
"""Patch LIBERO's __init__.py to skip interactive prompts in containers.

LIBERO's libero/libero/__init__.py has 3 input() calls that prompt the user
for a dataset directory. In non-interactive environments (Docker, Modal),
these cause EOFError. This script replaces them with defaults.
"""
import pathlib
import sys

target = pathlib.Path("/opt/LIBERO/libero/libero/__init__.py")
if not target.exists():
    print(f"ERROR: {target} not found")
    sys.exit(1)

import re

# Work on the full text (not line-by-line) to handle multi-line input() calls
text = target.read_text()

# Pattern 1: multi-line input(\n    "prompt"\n).lower()
text, n1 = re.subn(
    r'input\(\s*\n\s*"[^"]*"\s*\n\s*\)\.lower\(\)',
    '"n"',
    text,
)

# Pattern 2: multi-line input(\n    "prompt"\n)
text, n2 = re.subn(
    r'input\(\s*\n\s*"[^"]*"\s*\n\s*\)',
    '"/tmp/libero_data"',
    text,
)

# Pattern 3: single-line input().lower()
text, n3 = re.subn(r'input\(\)\.lower\(\)', '"n"', text)

# Pattern 4: single-line input("prompt")
text, n4 = re.subn(r'input\([^)]*\)', '"n"', text)

# Pattern 5: bare input()
text, n5 = re.subn(r'input\(\)', '"n"', text)

total = n1 + n2 + n3 + n4 + n5
target.write_text(text)
print(f"Patched {total} input() calls ({n1} multi-line+lower, {n2} multi-line, {n3} single+lower, {n4} single, {n5} bare)")

# Nuke .pyc caches
import subprocess
subprocess.run(["find", "/opt/LIBERO", "-name", "*.pyc", "-delete"], check=False)
subprocess.run(["find", "/opt/LIBERO", "-name", "__pycache__", "-type", "d", "-exec", "rm", "-rf", "{}", "+"], check=False)
print("Cleared .pyc caches")
