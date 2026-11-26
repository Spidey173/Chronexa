PYTHON ?= .venv/bin/python
PYTHONPATH := eval_harness:.

.PHONY: run test lint format eval clean

run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) src/server.py

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/ -v

lint:
	ruff check src tests eval eval_harness api

format:
	black --check src tests api

eval:
	./run.sh eval

