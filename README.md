# 🧹 Code Cleanup Automator

A chill Python toolkit that keeps your code fresh — refactors messy stuff, finds duplicates, and auto-writes tests so you don't have to cry over tech debt.

## 🧰 What's Inside

**`src/refactor-kit/`**

* `duplicate_finder.py` → spots copy-pasted duplicate code (uses AST + hashes)
* `event_dispatcher.py` → turns your if-elif jungle into a dispatch dictionary
* `test_generator.py` → whips up pytest tests with mocks, edge cases, and sass
* `lazy_loader.py` → makes imports chill until you actually need them
* `complexity_analyzer.sh` → checks how cursed your code complexity really is

## ⚙️ Tech Stack

Python 3.13+ with the essentials:

* `pytest`, `ruff`, `radon`, `pre-commit`
* `pipreqs` for keeping deps tight

## 🚀 Quick Start

```bash
# Find duplicates like a detective
python src/refactor-kit/duplicate_finder.py

# Auto-generate tests (because manual is ✨mid✨)
python src/refactor-kit/test_generator.py <source_file.py> -o test_output.py

# Refactor demo — dispatch style
python src/refactor-kit/event_dispatcher.py
```

## 📝 Notes

Runs with `uv` for dependency vibes.
Tests are still cooking, so don't @ me yet.

<br>
