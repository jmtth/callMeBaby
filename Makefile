NAME : CallMeBaby

install:
	@echo "Installing CallMeBaby..."
	uv sync

run:
	@echo "Running CallMeBaby..."
	uv run python main.py

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
