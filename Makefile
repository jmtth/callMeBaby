NAME : CallMeBaby

help:
	@echo "Makefile for $(NAME)"
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  install      Install the project"
	@echo "  run          Run the project"
	@echo "  debug        Debug the project"
	@echo "  clean        Clean up temporary files"
	@echo "  lint         Lint the project"
	@echo "  lint-strict  Strict linting of the project"

install:
	@echo "Installing CallMeBaby..."
	uv sync

run:
	@echo "Running CallMeBaby..."
	uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json
# 	uv run python main.py

test:
	@echo "Running tests for CallMeBaby..."
	uv run pytest

coverage:
	@echo "Running tests with coverage for CallMeBaby..."
	uv run pytest --cov --cov-report=term-missing

debug:
	@echo "Debugging CallMeBaby..."
	uv run python -m pdb -m src

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__ .pytest_cache .mypy_cache

lint:
	@echo "Linting CallMeBaby..."
	uv run flake8 src tests vendor_llm_sdk/llm_sdk
	uv run mypy src

lint-strict:
	@echo "Strict linting CallMeBaby..."
	uv run flake8 src tests
	uv run mypy src --strict

local:
	export HF_HUB_OFFLINE=1
	export TRANSFORMERS_OFFLINE=1

unset-local:
	unset HF_HUB_OFFLINE
	unset TRANSFORMERS_OFFLINE

.PHONY: install run debug clean lint lint-strict local unset-local
