"""Print the executable and installed package version."""

import sys

import bugclassinet

print(f"Python: {sys.version}")
print(f"BugClassiNet: {bugclassinet.__version__}")
