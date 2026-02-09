FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.7.1
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency definition
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --only main

# Copy application code
COPY . .

# --- THE FIX IS HERE ---
# Add 'src' to the Python Path so imports like 'from Moksha_1...' work
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Run the scheduler
CMD ["python", "src/Moksha_1/main_loop.py"]