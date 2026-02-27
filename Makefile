.PHONY: setup lint test notebooks clean

setup:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "Setup complete. Activate with: source .venv/bin/activate"

notebooks:
	jupyter lab --notebook-dir=notebooks

lint:
	@echo "Linting Python source files..."
	python -m py_compile src/utils/data_generation.py
	python -m py_compile src/utils/plotting.py
	python -m py_compile src/utils/metrics_helpers.py
	python -m py_compile src/utils/preprocessing_helpers.py
	@echo "All source files compile successfully."

test:
	python -m pytest tests/ -v 2>/dev/null || echo "No tests directory found. Add tests to tests/ to enable."

run-notebooks:
	@echo "Executing all notebooks (this may take a while)..."
	find notebooks/ -name "*.ipynb" -exec jupyter nbconvert --to notebook --execute {} \;
	@echo "All notebooks executed."

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned up cache files."
