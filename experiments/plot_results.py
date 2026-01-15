"""
Results Plotting Script

This script generates publication-ready plots from the experiment results.
All figures are saved to results/figures/ directory.

Generated Figures:
1. fig1_step_response.png - Step response comparison
2. fig2_disturbance.png - Disturbance rejection comparison
3. fig3_setpoint_tracking.png - Multi-setpoint tracking
4. fig4_mf_comparison.png - Membership function comparison
5. fig5_gains_evolution.png - Fuzzy PID gains over time
6. fig6_metrics_bar.png - Performance metrics bar chart
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.controllers.pid_controller import PIDController
from src.controllers.fuzzy_pid_controller import FuzzyPIDController
from src.simulation.temperature_system import (
    TemperatureSystem, pulse_disturbance
)
from src.simulation.simulator import Simulator, create_setpoint_profile

# Set matplotlib style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def ensure_dirs():
    """Create output directories if they don't exist."""
    results_dir = project_root / 'results'
    figures_dir = results_dir / 'figures'
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    return results_dir, figures_dir


def run_simulation(system, controller, setpoint_func, duration, name):
    """Helper function to run a simulation and return results."""
    simulator = Simulator(system, dt=0.1)
    return simulator.run(controller, setpoint_func, duration, name)


def plot_step_response(figures_dir):
    """
    Figure 1: Step Response Comparison
    
    Plots the temperature response of PID vs Fuzzy PID for a step
    change from 25C to 50C.
    """
    print("Generating Figure 1: Step Response...")
    
    system = TemperatureSystem(
        time_constant=50.0, gain=0.5, ambient_temp=20.0,
        initial_temp=25.0, dt=0.1, noise_std=0.1
    )
    
    pid = PIDController(kp=5.0, ki=0.1, kd=1.0, dt=0.1)
    fuzzy_pid = FuzzyPIDController(
        base_kp=5.0, base_ki=0.1, base_kd=1.0, dt=0.1,
        error_range=(-30, 30), delta_error_range=(-10, 10),
        mf_type='triangular'
    )
    
    setpoint_profile = create_setpoint_profile([(0.0, 25.0), (10.0, 50.0)])
    
    system.reset(25.0)
    pid_result = run_simulation(system, pid, setpoint_profile, 150.0, 'PID')
    
    system.reset(25.0)
    fuzzy_result = run_simulation(system, fuzzy_pid, setpoint_profile, 150.0, 'Fuzzy PID')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    # Temperature response
    ax1 = axes[0]
    ax1.plot(pid_result.time, pid_result.temperature, 'b-', 
             label='PID', linewidth=1.5)
    ax1.plot(fuzzy_result.time, fuzzy_result.temperature, 'r-', 
             label='Fuzzy PID', linewidth=1.5)
    ax1.plot(pid_result.time, pid_result.setpoint, 'k--', 
             label='Setpoint', linewidth=1.0, alpha=0.7)
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Step Response: PID vs Fuzzy PID Controller')
    ax1.legend(loc='lower right')
    ax1.set_ylim([20, 60])
    ax1.grid(True, alpha=0.3)
    
    # Control output
    ax2 = axes[1]
    ax2.plot(pid_result.time, pid_result.control_output, 'b-', 
             label='PID', linewidth=1.5)
    ax2.plot(fuzzy_result.time, fuzzy_result.control_output, 'r-', 
             label='Fuzzy PID', linewidth=1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Control Output (%)')
    ax2.legend(loc='upper right')
    ax2.set_ylim([-5, 105])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = figures_dir / 'fig1_step_response.png'
    plt.savefig(filepath)
    plt.close()
    print(f"  Saved: {filepath}")
    
    return pid_result, fuzzy_result


def plot_disturbance_rejection(figures_dir):
    """
    Figure 2: Disturbance Rejection Comparison
    
    Plots how each controller handles an external disturbance
    while maintaining the setpoint.
    """
    print("Generating Figure 2: Disturbance Rejection...")
    
    system = TemperatureSystem(
        time_constant=50.0, gain=0.5, ambient_temp=20.0,
        initial_temp=50.0, dt=0.1, noise_std=0.1
    )
    
    def disturbance(t):
        return pulse_disturbance(t, start_time=50.0, duration=20.0, magnitude=5.0)
    
    pid = PIDController(kp=5.0, ki=0.1, kd=1.0, dt=0.1)
    fuzzy_pid = FuzzyPIDController(
        base_kp=5.0, base_ki=0.1, base_kd=1.0, dt=0.1,
        error_range=(-30, 30), delta_error_range=(-10, 10),
        mf_type='triangular'
    )
    
    system.reset(50.0)
    system.set_disturbance(disturbance)
    pid_result = run_simulation(system, pid, 50.0, 150.0, 'PID')
    
    system.reset(50.0)
    system.set_disturbance(disturbance)
    fuzzy_result = run_simulation(system, fuzzy_pid, 50.0, 150.0, 'Fuzzy PID')
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Temperature response
    ax1 = axes[0]
    ax1.plot(pid_result.time, pid_result.temperature, 'b-', 
             label='PID', linewidth=1.5)
    ax1.plot(fuzzy_result.time, fuzzy_result.temperature, 'r-', 
             label='Fuzzy PID', linewidth=1.5)
    ax1.axhline(y=50, color='k', linestyle='--', alpha=0.7, label='Setpoint')
    ax1.axvspan(50, 70, alpha=0.2, color='yellow', label='Disturbance Period')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Disturbance Rejection: PID vs Fuzzy PID Controller')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Error
    ax2 = axes[1]
    ax2.plot(pid_result.time, pid_result.error, 'b-', 
             label='PID', linewidth=1.5)
    ax2.plot(fuzzy_result.time, fuzzy_result.error, 'r-', 
             label='Fuzzy PID', linewidth=1.5)
    ax2.axvspan(50, 70, alpha=0.2, color='yellow')
    ax2.set_ylabel('Error (°C)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Control output
    ax3 = axes[2]
    ax3.plot(pid_result.time, pid_result.control_output, 'b-', 
             label='PID', linewidth=1.5)
    ax3.plot(fuzzy_result.time, fuzzy_result.control_output, 'r-', 
             label='Fuzzy PID', linewidth=1.5)
    ax3.axvspan(50, 70, alpha=0.2, color='yellow')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Control Output (%)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = figures_dir / 'fig2_disturbance_rejection.png'
    plt.savefig(filepath)
    plt.close()
    print(f"  Saved: {filepath}")
    
    return pid_result, fuzzy_result


def plot_setpoint_tracking(figures_dir):
    """
    Figure 3: Setpoint Tracking Comparison
    
    Plots the response to multiple setpoint changes.
    """
    print("Generating Figure 3: Setpoint Tracking...")
    
    system = TemperatureSystem(
        time_constant=50.0, gain=0.5, ambient_temp=20.0,
        initial_temp=25.0, dt=0.1, noise_std=0.1
    )
    
    pid = PIDController(kp=5.0, ki=0.1, kd=1.0, dt=0.1)
    fuzzy_pid = FuzzyPIDController(
        base_kp=5.0, base_ki=0.1, base_kd=1.0, dt=0.1,
        error_range=(-40, 40), delta_error_range=(-15, 15),
        mf_type='triangular'
    )
    
    setpoint_profile = create_setpoint_profile([
        (0.0, 25.0), (50.0, 50.0), (120.0, 35.0), (190.0, 60.0)
    ])
    
    system.reset(25.0)
    pid_result = run_simulation(system, pid, setpoint_profile, 280.0, 'PID')
    
    system.reset(25.0)
    fuzzy_result = run_simulation(system, fuzzy_pid, setpoint_profile, 280.0, 'Fuzzy PID')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(pid_result.time, pid_result.temperature, 'b-', 
            label='PID', linewidth=1.5)
    ax.plot(fuzzy_result.time, fuzzy_result.temperature, 'r-', 
            label='Fuzzy PID', linewidth=1.5)
    ax.plot(pid_result.time, pid_result.setpoint, 'k--', 
            label='Setpoint', linewidth=2.0, alpha=0.7)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Setpoint Tracking: PID vs Fuzzy PID Controller')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = figures_dir / 'fig3_setpoint_tracking.png'
    plt.savefig(filepath)
    plt.close()
    print(f"  Saved: {filepath}")
    
    return pid_result, fuzzy_result


def plot_mf_comparison(figures_dir):
    """
    Figure 4: Membership Function Comparison
    
    Compares Triangular vs Gaussian membership functions for Fuzzy PID.
    """
    print("Generating Figure 4: Membership Function Comparison...")
    
    system = TemperatureSystem(
        time_constant=50.0, gain=0.5, ambient_temp=20.0,
        initial_temp=25.0, dt=0.1, noise_std=0.1
    )
    
    fuzzy_tri = FuzzyPIDController(
        base_kp=5.0, base_ki=0.1, base_kd=1.0, dt=0.1,
        error_range=(-30, 30), delta_error_range=(-10, 10),
        mf_type='triangular'
    )
    
    fuzzy_gauss = FuzzyPIDController(
        base_kp=5.0, base_ki=0.1, base_kd=1.0, dt=0.1,
        error_range=(-30, 30), delta_error_range=(-10, 10),
        mf_type='gaussian'
    )
    
    setpoint_profile = create_setpoint_profile([(0.0, 25.0), (10.0, 50.0)])
    
    system.reset(25.0)
    tri_result = run_simulation(system, fuzzy_tri, setpoint_profile, 150.0, 'Triangular')
    
    system.reset(25.0)
    gauss_result = run_simulation(system, fuzzy_gauss, setpoint_profile, 150.0, 'Gaussian')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(tri_result.time, tri_result.temperature, 'b-', 
            label='Triangular MFs', linewidth=1.5)
    ax.plot(gauss_result.time, gauss_result.temperature, 'g-', 
            label='Gaussian MFs', linewidth=1.5)
    ax.plot(tri_result.time, tri_result.setpoint, 'k--', 
            label='Setpoint', linewidth=1.0, alpha=0.7)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Fuzzy PID: Triangular vs Gaussian Membership Functions')
    ax.legend(loc='lower right')
    ax.set_ylim([20, 60])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = figures_dir / 'fig4_mf_comparison.png'
    plt.savefig(filepath)
    plt.close()
    print(f"  Saved: {filepath}")
    
    return tri_result, gauss_result


def plot_gains_evolution(figures_dir):
    """
    Figure 5: Fuzzy PID Gains Evolution
    
    Shows how the Kp, Ki, Kd gains change over time in the Fuzzy PID controller.
    """
    print("Generating Figure 5: Gains Evolution...")
    
    system = TemperatureSystem(
        time_constant=50.0, gain=0.5, ambient_temp=20.0,
        initial_temp=25.0, dt=0.1, noise_std=0.1
    )
    
    fuzzy_pid = FuzzyPIDController(
        base_kp=5.0, base_ki=0.1, base_kd=1.0, dt=0.1,
        error_range=(-30, 30), delta_error_range=(-10, 10),
        mf_type='triangular'
    )
    
    setpoint_profile = create_setpoint_profile([(0.0, 25.0), (10.0, 50.0)])
    
    system.reset(25.0)
    result = run_simulation(system, fuzzy_pid, setpoint_profile, 150.0, 'Fuzzy PID')
    
    if result.gains_history is None:
        print("  Warning: Gains history not available")
        return None
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    
    # Temperature
    ax1 = axes[0]
    ax1.plot(result.time, result.temperature, 'b-', linewidth=1.5)
    ax1.plot(result.time, result.setpoint, 'k--', alpha=0.7)
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Fuzzy PID Controller: Adaptive Gain Evolution')
    ax1.grid(True, alpha=0.3)
    
    # Kp
    ax2 = axes[1]
    ax2.plot(result.time, result.gains_history['Kp'], 'r-', linewidth=1.5)
    ax2.axhline(y=5.0, color='k', linestyle='--', alpha=0.5, label='Base Kp')
    ax2.set_ylabel('Kp')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Ki
    ax3 = axes[2]
    ax3.plot(result.time, result.gains_history['Ki'], 'g-', linewidth=1.5)
    ax3.axhline(y=0.1, color='k', linestyle='--', alpha=0.5, label='Base Ki')
    ax3.set_ylabel('Ki')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Kd
    ax4 = axes[3]
    ax4.plot(result.time, result.gains_history['Kd'], 'm-', linewidth=1.5)
    ax4.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Base Kd')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Kd')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = figures_dir / 'fig5_gains_evolution.png'
    plt.savefig(filepath)
    plt.close()
    print(f"  Saved: {filepath}")
    
    return result


def plot_metrics_comparison(figures_dir, pid_result, fuzzy_result):
    """
    Figure 6: Performance Metrics Bar Chart
    
    Visual comparison of key performance metrics between controllers.
    """
    print("Generating Figure 6: Metrics Comparison...")
    
    metrics = ['IAE', 'ISE', 'Rise Time\n(s)', 'Settling Time\n(s)', 
               'Overshoot\n(%)', 'SS Error\n(°C)']
    
    pid_values = [
        pid_result.iae,
        pid_result.ise,
        pid_result.rise_time if pid_result.rise_time != float('inf') else 0,
        pid_result.settling_time if pid_result.settling_time != float('inf') else 0,
        pid_result.overshoot,
        pid_result.steady_state_error
    ]
    
    fuzzy_values = [
        fuzzy_result.iae,
        fuzzy_result.ise,
        fuzzy_result.rise_time if fuzzy_result.rise_time != float('inf') else 0,
        fuzzy_result.settling_time if fuzzy_result.settling_time != float('inf') else 0,
        fuzzy_result.overshoot,
        fuzzy_result.steady_state_error
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    rects1 = ax.bar(x - width/2, pid_values, width, label='PID', color='steelblue')
    rects2 = ax.bar(x + width/2, fuzzy_values, width, label='Fuzzy PID', color='coral')
    
    ax.set_ylabel('Value')
    ax.set_title('Performance Metrics Comparison: PID vs Fuzzy PID')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    add_labels(rects1)
    add_labels(rects2)
    
    plt.tight_layout()
    filepath = figures_dir / 'fig6_metrics_comparison.png'
    plt.savefig(filepath)
    plt.close()
    print(f"  Saved: {filepath}")


def plot_membership_functions(figures_dir):
    """
    Figure 7: Membership Functions Visualization
    
    Shows the membership functions used in the Fuzzy PID controller.
    """
    print("Generating Figure 7: Membership Functions...")
    
    from src.fuzzy_logic.fuzzy_sets import create_linguistic_variable
    
    # Create membership functions
    error_mfs = create_linguistic_variable('error', -30, 30, 5, 'triangular')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    
    x = np.linspace(-30, 30, 500)
    colors = ['blue', 'green', 'gray', 'orange', 'red']
    labels = ['NB (Negative Big)', 'NS (Negative Small)', 'ZE (Zero)', 
              'PS (Positive Small)', 'PB (Positive Big)']
    
    for (name, mf), color, label in zip(error_mfs.items(), colors, labels):
        y = [mf.compute(xi) for xi in x]
        ax.plot(x, y, color=color, linewidth=2, label=label)
    
    ax.set_xlabel('Error (°C)')
    ax.set_ylabel('Membership Degree')
    ax.set_title('Triangular Membership Functions for Error Variable')
    ax.legend(loc='upper right')
    ax.set_ylim([-0.05, 1.1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = figures_dir / 'fig7_membership_functions.png'
    plt.savefig(filepath)
    plt.close()
    print(f"  Saved: {filepath}")


def main():
    """Generate all publication-ready plots."""
    print("\n" + "#" * 70)
    print("# GENERATING PUBLICATION-READY FIGURES")
    print("#" * 70)
    
    _, figures_dir = ensure_dirs()
    
    # Generate all figures
    pid_result, fuzzy_result = plot_step_response(figures_dir)
    plot_disturbance_rejection(figures_dir)
    plot_setpoint_tracking(figures_dir)
    plot_mf_comparison(figures_dir)
    plot_gains_evolution(figures_dir)
    plot_metrics_comparison(figures_dir, pid_result, fuzzy_result)
    plot_membership_functions(figures_dir)
    
    print("\n" + "#" * 70)
    print("# ALL FIGURES GENERATED SUCCESSFULLY")
    print(f"# Figures saved to: {figures_dir}")
    print("#" * 70)
    
    # Print figure list for user reference
    print("\nGenerated figures:")
    print("  1. fig1_step_response.png - Step response comparison (for Section: Results)")
    print("  2. fig2_disturbance_rejection.png - Disturbance test (for Section: Results)")
    print("  3. fig3_setpoint_tracking.png - Multi-setpoint tracking (for Section: Results)")
    print("  4. fig4_mf_comparison.png - MF comparison (for Section: Discussion)")
    print("  5. fig5_gains_evolution.png - Adaptive gains (for Section: Methodology)")
    print("  6. fig6_metrics_comparison.png - Metrics bar chart (for Section: Results)")
    print("  7. fig7_membership_functions.png - MF visualization (for Section: Methodology)")


if __name__ == '__main__':
    main()
