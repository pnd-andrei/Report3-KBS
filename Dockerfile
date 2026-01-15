FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY experiments/ ./experiments/

# Create results directory
RUN mkdir -p results/figures

# Default command: run experiments and generate plots
CMD ["sh", "-c", "python experiments/run_experiments.py && python experiments/plot_results.py"]
