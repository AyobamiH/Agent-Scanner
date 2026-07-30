"""String utility functions.

This module is reserved for general-purpose string manipulation utilities
that may be used across multiple detector modules. Currently, complex string
processing (such as tokenisation) is integrated directly into the modules
that use them to maintain cohesion and avoid over-abstraction.

In the future, if common string utilities are needed (e.g., safe truncation,
normalized splitting, pattern validation), they should be added here with
clear docstrings and type hints.
"""

__all__: list[str] = []
