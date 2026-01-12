#!/bin/bash
set -e  # Exit on any error

echo "=== Moksha 2.0 Clean Installation ==="
echo ""

# Navigate to project
cd ~/Documents/Patience/Analysis/TARS/MOKSHA/Moksha_1

echo "Step 1: Cleaning old environment..."
rm -rf .venv poetry.lock
rm -rf __pycache__ .pytest_cache .mypy_cache

echo "Step 2: Verifying Python version..."
python --version
if ! python --version | grep -q "3.10"; then
    echo "ERROR: Python 3.10 not active. Run: pyenv global 3.10.13"
    exit 1
fi

echo "Step 3: Clearing Poetry cache..."
poetry cache clear pypi --all --no-interaction

echo "Step 4: Generating new lock file..."
poetry lock -vv

echo "Step 5: Installing dependencies..."
poetry install --no-root -vv

echo "Step 6: Installing project..."
poetry install -vv

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Verify installation:"
echo "  poetry shell"
echo "  python -c 'import pandas; import alpaca; print(\"Success!\")'"