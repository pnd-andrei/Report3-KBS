# Fuzzy PID Temperature Controller

A Python implementation of a Fuzzy PID Controller for temperature control systems, comparing its performance against a conventional PID controller.

**Repository:** https://github.com/pnd-andrei/Report3-KBS

## Project Overview

This project implements and evaluates a **Fuzzy PID Controller** for temperature control, building upon the theoretical foundations of fuzzy logic and systematic literature review findings that identified temperature control as the most common application area for Fuzzy Logic Controllers (FLCs) in industrial settings.

### Research Questions

- **RQ1**: How does a Fuzzy PID controller perform compared to a conventional PID controller in temperature control scenarios?
- **RQ2**: What is the impact of different fuzzy membership function shapes on controller performance?
- **RQ3**: How do the controllers perform under different disturbance conditions?

## Repository Structure

```
Report3-KBS/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker container definition
├── src/
│   ├── fuzzy_logic/          # Core fuzzy logic implementation
│   │   ├── fuzzy_sets.py     # Membership functions
│   │   ├── fuzzy_rules.py    # Rule base
│   │   └── inference_engine.py
│   ├── controllers/          # Controller implementations
│   │   ├── pid_controller.py
│   │   └── fuzzy_pid_controller.py
│   └── simulation/           # Simulation components
│       ├── temperature_system.py
│       └── simulator.py
├── experiments/              # Experiment scripts
│   ├── run_experiments.py
│   └── plot_results.py
└── results/                  # Output directory
    └── figures/              # Generated plots
```

## Installation

### Option 1: Using Docker (Recommended)

Build and run with Docker:

```bash
# Build the Docker image
docker build -t fuzzy-pid-controller .

# Run experiments and generate plots
docker run -v $(pwd)/results:/app/results fuzzy-pid-controller

# On Windows PowerShell:
docker run -v ${PWD}/results:/app/results fuzzy-pid-controller
```

Results will be saved to the `results/` directory.

### Option 2: Local Installation

Prerequisites:
- Python 3.8 or higher
- pip package manager

Setup:

```bash
# Clone the repository
git clone https://github.com/pnd-andrei/Report3-KBS.git
cd Report3-KBS

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Experiments

### Using Docker

```bash
docker run -v $(pwd)/results:/app/results fuzzy-pid-controller
```

### Using Local Installation

```bash
# Run all experiments
python experiments/run_experiments.py

# Generate plots
python experiments/plot_results.py
```

### Experiments Included

1. **Step Response Test**: Setpoint change from 25°C to 50°C
2. **Disturbance Rejection Test**: External disturbance handling
3. **Setpoint Tracking Test**: Multiple setpoint changes
4. **Membership Function Comparison**: Triangular vs Gaussian MFs

### Output Files

After running experiments:

| File | Description |
|------|-------------|
| `results/all_metrics.csv` | Performance metrics summary |
| `results/exp1_step_response.csv` | Step response data |
| `results/exp2_disturbance.csv` | Disturbance test data |
| `results/exp3_setpoint_tracking.csv` | Tracking data |
| `results/exp4_mf_comparison.csv` | MF comparison data |
| `results/figures/*.png` | Generated plots |

## Performance Metrics

| Metric | Description |
|--------|-------------|
| **IAE** | Integral of Absolute Error |
| **ISE** | Integral of Squared Error |
| **ITAE** | Integral of Time-weighted Absolute Error |
| **Rise Time** | Time to reach 90% of setpoint |
| **Settling Time** | Time to stay within 2% of setpoint |
| **Overshoot** | Maximum percentage above setpoint |

## Implementation Details

### Fuzzy Logic Engine

- **Membership Functions**: Triangular, Trapezoidal, Gaussian
- **Inference Method**: Mamdani
- **Defuzzification**: Center of Gravity (COG)

### Fuzzy PID Controller

- **Inputs**: Error (e), Change in Error (de)
- **Outputs**: Adjustment factors for Kp, Ki, Kd
- **Rule Base**: 25 rules (5x5 grid)
- **Linguistic Terms**: NB, NS, ZE, PS, PB

### Temperature System Model

First-order thermal system with:
- Configurable time constant (default: 50s)
- Process gain (default: 0.5)
- External disturbance support
- Measurement noise simulation

## Results Summary

The Fuzzy PID controller demonstrates improvements over conventional PID:

| Metric | Improvement |
|--------|-------------|
| IAE | ~28% reduction |
| Overshoot | ~34% reduction |
| Settling Time | ~22% faster |
| Disturbance Rejection | ~31% better |

## License

This project is developed for academic purposes as part of the Knowledge-Based Systems course.

## Author

Andrei Panduru - Artificial Intelligence, UBB

## References

1. Zadeh, L. A. (1965). "Fuzzy sets." Information and Control, 8(3), 338-353.
2. Mamdani, E. H., & Assilian, S. (1975). "An experiment in linguistic synthesis with a fuzzy logic controller."
3. Lee, C. C. (1990). "Fuzzy logic in control systems." IEEE Transactions on Systems, Man, and Cybernetics.
