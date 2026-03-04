# Contributing to Battousai

Thank you for your interest in contributing to Battousai!

## Development Environment

**Requirements:** Python 3.10 or later. No external dependencies.

```bash
git clone https://github.com/DPL1979/battousai.git
cd battousai
python --version  # confirm 3.10+
```

## Running the Tests

```bash
python -m unittest discover -s tests -v
```

## Code Style

- Type hints everywhere
- Docstrings on public classes and methods
- Pure Python, no external dependencies
- No global mutable state outside the Kernel object
