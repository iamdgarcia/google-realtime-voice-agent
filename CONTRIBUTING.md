# Contributing Guide

Thanks for your interest in improving this project.

## Local setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run checks

```bash
pytest -q
./venv/bin/python -m py_compile server.py
```

## Development rules

- Keep changes focused and small.
- Add or update tests when behavior changes.
- Prefer dependency injection over hard-wired adapters.
- Document any new environment variables in `README.md` and `config.py`.

## Pull requests

- Use a clear title and problem statement.
- Include what changed, why, and how it was validated.
- Link related issue(s).
- Include logs/screenshots when UI or runtime behavior changes.
