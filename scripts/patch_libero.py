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

lines = target.read_text().splitlines()
patched = 0

# Dump lines 60-80 for debugging
print("--- Lines 60-80 BEFORE patch ---")
for i in range(max(0, 59), min(len(lines), 80)):
    print(f"  {i+1:3d}: {lines[i]!r}")

new_lines = []
for i, line in enumerate(lines):
    if "input(" in line or "input()" in line:
        # Replace ANY input() or input("...") with a safe default
        import re
        new_line = re.sub(r'input\([^)]*\)\.lower\(\)', '"n"', line)
        new_line = re.sub(r'input\([^)]*\)', '"n"', new_line)
        new_line = re.sub(r'input\(\)\.lower\(\)', '"n"', new_line)
        new_line = re.sub(r'input\(\)', '"n"', new_line)
        if new_line != line:
            patched += 1
            print(f"  Line {i+1}: {line.strip()!r} -> {new_line.strip()!r}")
        new_lines.append(new_line)
    else:
        new_lines.append(line)

target.write_text("\n".join(new_lines) + "\n")
print(f"Patched {patched} input() calls in {target}")

# Nuke all .pyc caches so Python doesn't use stale bytecode
import subprocess
subprocess.run(["find", "/opt/LIBERO", "-name", "*.pyc", "-delete"], check=False)
subprocess.run(["find", "/opt/LIBERO", "-name", "__pycache__", "-type", "d", "-exec", "rm", "-rf", "{}", "+"], check=False)
print("Cleared .pyc caches")
