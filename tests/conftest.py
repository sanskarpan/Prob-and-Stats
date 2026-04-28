"""
Pytest configuration: add the package root to sys.path so that test modules
can import the Prob&Stats modules directly (without package installation).
"""
import sys
import os

# Insert the parent directory of 'tests/' (i.e. the Prob&Stats root) at the
# front of sys.path so that `import probability`, `import distributions`, etc.
# all resolve to the local source files.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
