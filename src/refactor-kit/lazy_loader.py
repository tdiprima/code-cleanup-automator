# This script shows how to use importlib for lazy loading of modules,
# reducing startup time by importing only when needed. Assumes a module
# like 'heavy_module' with a 'run' function exists.

import importlib


def lazy_import(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        print(f"Module '{module_name}' not found.")
        return None


# Example usage
if __name__ == "__main__":
    heavy_tool = lazy_import("heavy_module")
    if heavy_tool:
        heavy_tool.run()
