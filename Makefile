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
	uv run python main.py

test:
	@echo "Running tests for CallMeBaby..."
	uv run pytest

coverage:
	@echo "Running tests with coverage for CallMeBaby..."
	uv run pytest --cov

debug:
	@echo "Debugging CallMeBaby..."
	uv run python -m pdb main.py

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__ .pytest_cache .mypy_cache

lint:
	@echo "Linting CallMeBaby..."
	uv run flake8 ./src
	uv run mypy ./src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	@echo "Strict linting CallMeBaby..."
	uv run flake8 ./src
	uv run mypy ./src --strict

.PHONY: install run debug clean lint lint-strict
