import importlib
import inspect
import pkgutil
from typing import get_type_hints

import nano_inference
import pytest


def get_all_modules(package):
    """Recursively find all modules in a package."""
    modules = []
    if hasattr(package, "__path__"):
        for loader, module_name, is_pkg in pkgutil.walk_packages(
            package.__path__, package.__name__ + "."
        ):
            try:
                modules.append(importlib.import_module(module_name))
            except ImportError:
                continue
    return modules


@pytest.mark.parametrize("module", get_all_modules(nano_inference))
def test_type_hints_resolve(module):
    """
    Force resolution of all type hints in a module.
    This catches missing imports used in type annotations that are normally
    deferred by Python 3.10+ or from __future__ import annotations.
    """
    # 1. Check all classes in the module
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__:
            try:
                # get_type_hints() forces resolution of all annotations
                get_type_hints(obj)

                # Also check all methods in the class
                for attr_name, attr_obj in inspect.getmembers(obj, inspect.isfunction):
                    get_type_hints(attr_obj)
            except NameError as e:
                pytest.fail(
                    f"Type hint resolution failed in class {module.__name__}.{name}: {e}"
                )
            except Exception:
                # Some objects might fail for other reasons, skip them
                continue

    # 2. Check all functions in the module
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if obj.__module__ == module.__name__:
            try:
                get_type_hints(obj)
            except NameError as e:
                pytest.fail(
                    f"Type hint resolution failed in function {module.__name__}.{name}: {e}"
                )
            except Exception:
                continue
