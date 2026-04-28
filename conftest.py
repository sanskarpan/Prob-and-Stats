"""
Top-level pytest conftest for the Prob&Stats module.

Prevents pytest from collecting __init__.py (which uses relative imports
and cannot be imported as a standalone file), and adds the module root to
sys.path so test files can use direct module imports.
"""
import sys
import os

# Exclude __init__.py from test collection — it uses relative imports
# and is not a test file.
collect_ignore = ["__init__.py"]

# Ensure the Prob&Stats directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
