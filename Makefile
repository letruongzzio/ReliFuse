.PHONY: install test lint format-check demo

install:
	python -m pip install -e ".[demo,dev]"

test:
	python -m pytest

lint:
	python -m ruff check src/relifuse tests examples

format-check:
	python -m ruff format --check src/relifuse tests examples

demo:
	python examples/two_expert_demo.py
