"""
Experiment Runner

This script runs all experimental scenarios comparing the Fuzzy PID
controller with the conventional PID controller for temperature control.

Experiments:
1. Step Response: Setpoint change from 25C to 50C
2. Disturbance Rejection: External disturbance handling
3. Setpoint Tracking: Multiple setpoint changes
4. Membership Function Comparison: Triangular vs Gaussian MFs

Results are saved to the results/ directory.
"""

import sys
import os
import numpy as np
import csv
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.controllers.pid_controller import PIDController
from src.controllers.fuzzy_pid_controller import FuzzyPIDController
from src.simulation.temperature_system import (
    TemperatureSystem, step_disturbance, pulse_disturbance
)
from src.simulation.simulator import (
    Simulator, SimulationResult, create_setpoint_profile, 
    print_metrics_comparison
)


def ensure_dirs():
    """Create output directories if they don't exist."""
    results_dir = project_root / 'results'
    figures_dir = results_dir / 'figures'
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    return results_dir, figures_dir


def save_results_csv(results: dict, filename: str, results_dir: Path):
    """Save simulation results to CSV file."""
    filepath = results_dir / filename
    
    # Get column headers
    controller_names = list(results.keys())
    
    # Find max length
    max_len = max(len(r.time) for r in results.values())
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['time']
        for name in controller_names:
            header.extend([f'{name}_temp', f'{name}_setpoint', 
                          f'{name}_control', f'{name}_error'])
        writer.writerow(header)
        
        # Data rows
        for i in range(max_len):
            row = []
            for j, name in enumerate(controller_names):
                r = results[name]
                if i < len(r.time):
                    if j == 0:
                        row.append(r.time[i])
                    row.extend([r.temperature[i], r.setpoint[i], 
                               r.control_output[i], r.error[i]])
                else:
                    if j == 0:
                        row.append('')
                    row.extend(['', '', '', ''])
            writer.writerow(row)
    
    print(f"Saved: {filepath}")


def save_metrics_csv(all_metrics: dict, filename: str, results_dir: Path):
    """Save performance metrics to CSV file."""
    filepath = results_dir / filename
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['Experiment', 'Controller', 'IAE', 'ISE', 'ITAE', 
                 'Rise_Time', 'Settling_Time', 'Overshoot', 'SS_Error']
        writer.writerow(header)
        
        # Data
        for exp_name, results in all_metrics.items():
            for ctrl_name, r in results.items():
                row = [
                    exp_name, ctrl_name,
                    f"{r.iae:.4f}",
                    f"{r.ise:.4f}",
                    f"{r.itae:.4f}",
                    f"{r.rise_time:.4f}" if r.rise_time != float('inf') else 'N/A',
                    f"{r.settling_time:.4f}" if r.settling_time != float('inf') else 'N/A',
                    f"{r.overshoot:.4f}",
                    f"{r.steady_state_error:.6f}"
                ]
                writer.writerow(row)
    
    print(f"Saved: {filepath}")


def experiment_step_response():
    """
    Experiment 1: Step Response Test
    
    Tests the response of both controllers to a step change in setpoint
    from 25C to 50C.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Step Response Test")
    print("Setpoint: 25C -> 50C at t=10s")
    print("=" * 70)
    
    # System parameters
    system = TemperatureSystem(
        time_constant=50.0,
        gain=0.5,
        ambient_temp=20.0,
        initial_temp=25.0,
        dt=0.1,
        noise_std=0.1
    )
    
    # Controllers
    pid = PIDController(kp=5.0, ki=0.1, kd=1.0, dt=0.1, 
                        output_min=0, output_max=100)
    
    fuzzy_pid = FuzzyPIDController(
        base_kp=5.0, base_ki=0.1, base_kd=1.0, dt=0.1,
        error_range=(-30, 30), delta_error_range=(-10, 10),
        output_min=0, output_max=100, mf_type='triangular'
    )
    
    # Setpoint profile
    setpoint_profile = create_setpoint_profile([
        (0.0, 25.0),
        (10.0, 50.0)
    ])
    
    # Run simulations
    simulator = Simulator(system, dt=0.1)
    
    system.reset(25.0)
    pid_result = simulator.run(pid, setpoint_profile, duration=150.0, 
                               controller_name='PID')
    
    system.reset(25.0)
    fuzzy_result = simulator.run(fuzzy_pid, setpoint_profile, duration=150.0,
                                 controller_name='Fuzzy PID')
    
    results = {'PID': pid_result, 'Fuzzy PID': fuzzy_result}
    print_metrics_comparison(results)
    
    return results


def experiment_disturbance_rejection():
    """
    Experiment 2: Disturbance Rejection Test
    
    Tests how well each controller rejects external disturbances
    while maintaining the setpoint.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Disturbance Rejection Test")
    print("Setpoint: 50C, Disturbance: +5 degrees/s at t=50s for 20s")
    print("=" * 70)
    
    system = TemperatureSystem(
        time_constant=50.0,
        gain=0.5,
        ambient_temp=20.0,
        initial_temp=50.0,
        dt=0.1,
        noise_std=0.1
    )
    
    # Add disturbance
    def disturbance(t):
        return pulse_disturbance(t, start_time=50.0, duration=20.0, magnitude=5.0)
    
    system.set_disturbance(disturbance)
    
    # Controllers
    pid = PIDController(kp=5.0, ki=0.1, kd=1.0, dt=0.1,
                        output_min=0, output_max=100)
    
    fuzzy_pid = FuzzyPIDController(
        base_kp=5.0, base_ki=0.1, base_kd=1.0, dt=0.1,
        error_range=(-30, 30), delta_error_range=(-10, 10),
        output_min=0, output_max=100, mf_type='triangular'
    )
    
    # Constant setpoint
    setpoint = 50.0
    
    simulator = Simulator(system, dt=0.1)
    
    system.reset(50.0)
    system.set_disturbance(disturbance)
    pid_result = simulator.run(pid, setpoint, duration=150.0,
                               controller_name='PID')
    
    system.reset(50.0)
    system.set_disturbance(disturbance)
    fuzzy_result = simulator.run(fuzzy_pid, setpoint, duration=150.0,
                                 controller_name='Fuzzy PID')
    
    results = {'PID': pid_result, 'Fuzzy PID': fuzzy_result}
    print_metrics_comparison(results)
    
    return results


def experiment_setpoint_tracking():
    """
    Experiment 3: Setpoint Tracking Test
    
    Tests how well each controller tracks multiple setpoint changes.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Setpoint Tracking Test")
    print("Setpoints: 25C -> 50C -> 35C -> 60C")
    print("=" * 70)
    
    system = TemperatureSystem(
        time_constant=50.0,
        gain=0.5,
        ambient_temp=20.0,
        initial_temp=25.0,
        dt=0.1,
        noise_std=0.1
    )
    
    # Controllers
    pid = PIDController(kp=5.0, ki=0.1, kd=1.0, dt=0.1,
                        output_min=0, output_max=100)
    
    fuzzy_pid = FuzzyPIDController(
        base_kp=5.0, base_ki=0.1, base_kd=1.0, dt=0.1,
        error_range=(-40, 40), delta_error_range=(-15, 15),
        output_min=0, output_max=100, mf_type='triangular'
    )
    
    # Multiple setpoint changes
    setpoint_profile = create_setpoint_profile([
        (0.0, 25.0),
        (50.0, 50.0),
        (120.0, 35.0),
        (190.0, 60.0)
    ])
    
    simulator = Simulator(system, dt=0.1)
    
    system.reset(25.0)
    pid_result = simulator.run(pid, setpoint_profile, duration=280.0,
                               controller_name='PID')
    
    system.reset(25.0)
    fuzzy_result = simulator.run(fuzzy_pid, setpoint_profile, duration=280.0,
                                 controller_name='Fuzzy PID')
    
    results = {'PID': pid_result, 'Fuzzy PID': fuzzy_result}
    print_metrics_comparison(results)
    
    return results


def experiment_mf_comparison():
    """
    Experiment 4: Membership Function Comparison
    
    Compares the performance of Fuzzy PID with different membership
    function types (Triangular vs Gaussian).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Membership Function Comparison")
    print("Comparing Triangular vs Gaussian MFs for Fuzzy PID")
    print("=" * 70)
    
    system = TemperatureSystem(
        time_constant=50.0,
        gain=0.5,
        ambient_temp=20.0,
        initial_temp=25.0,
        dt=0.1,
        noise_std=0.1
    )
    
    # Controllers with different MF types
    fuzzy_triangular = FuzzyPIDController(
        base_kp=5.0, base_ki=0.1, base_kd=1.0, dt=0.1,
        error_range=(-30, 30), delta_error_range=(-10, 10),
        output_min=0, output_max=100, mf_type='triangular'
    )
    
    fuzzy_gaussian = FuzzyPIDController(
        base_kp=5.0, base_ki=0.1, base_kd=1.0, dt=0.1,
        error_range=(-30, 30), delta_error_range=(-10, 10),
        output_min=0, output_max=100, mf_type='gaussian'
    )
    
    # Step response test
    setpoint_profile = create_setpoint_profile([
        (0.0, 25.0),
        (10.0, 50.0)
    ])
    
    simulator = Simulator(system, dt=0.1)
    
    system.reset(25.0)
    tri_result = simulator.run(fuzzy_triangular, setpoint_profile, 
                               duration=150.0, controller_name='Fuzzy (Triangular)')
    
    system.reset(25.0)
    gauss_result = simulator.run(fuzzy_gaussian, setpoint_profile, 
                                 duration=150.0, controller_name='Fuzzy (Gaussian)')
    
    results = {'Fuzzy (Triangular)': tri_result, 'Fuzzy (Gaussian)': gauss_result}
    print_metrics_comparison(results)
    
    return results


def main():
    """Run all experiments and save results."""
    print("\n" + "#" * 70)
    print("# FUZZY PID TEMPERATURE CONTROLLER - EXPERIMENTS")
    print("#" * 70)
    
    # Ensure output directories exist
    results_dir, figures_dir = ensure_dirs()
    
    # Run all experiments
    all_metrics = {}
    
    # Experiment 1: Step Response
    exp1_results = experiment_step_response()
    save_results_csv(exp1_results, 'exp1_step_response.csv', results_dir)
    all_metrics['Step Response'] = exp1_results
    
    # Experiment 2: Disturbance Rejection
    exp2_results = experiment_disturbance_rejection()
    save_results_csv(exp2_results, 'exp2_disturbance.csv', results_dir)
    all_metrics['Disturbance Rejection'] = exp2_results
    
    # Experiment 3: Setpoint Tracking
    exp3_results = experiment_setpoint_tracking()
    save_results_csv(exp3_results, 'exp3_setpoint_tracking.csv', results_dir)
    all_metrics['Setpoint Tracking'] = exp3_results
    
    # Experiment 4: MF Comparison
    exp4_results = experiment_mf_comparison()
    save_results_csv(exp4_results, 'exp4_mf_comparison.csv', results_dir)
    all_metrics['MF Comparison'] = exp4_results
    
    # Save combined metrics
    save_metrics_csv(all_metrics, 'all_metrics.csv', results_dir)
    
    print("\n" + "#" * 70)
    print("# ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print(f"# Results saved to: {results_dir}")
    print("#" * 70)
    
    return all_metrics


if __name__ == '__main__':
    main()
