# Use python 3.9 slim
FROM python:3.9-slim

WORKDIR /app

# Install system basics
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 1. Force-Install Critical Libraries Globally (Bypassing Poetry for stability)
# UPDATED: Streamlit 1.32.0 supports st.rerun() and new layout options
RUN pip install --no-cache-dir \
    pandas==2.0.3 \
    numpy==1.24.3 \
    alpaca-trade-api==3.0.2 \
    streamlit==1.32.0 \
    python-dotenv==1.0.0 \
    watchdog==3.0.0 \
    pytz

# 2. Copy the project code
COPY . .

# 3. Set Python Path so 'import Moksha_1' works everywhere
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Default command
CMD ["python", "src/production_equity.py"]