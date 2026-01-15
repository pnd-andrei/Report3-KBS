"""
Simulation Manager Module

This module provides the main simulation loop and data collection
for running temperature control experiments. It manages the interaction
between the controller and the temperature system, collecting data
for analysis and visualization.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .temperature_system import TemperatureSystem


@dataclass
class SimulationResult:
    """
    Container for simulation results.
    
    Stores all time series data and computed performance metrics
    from a simulation run.
    """
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    temperature: np.ndarray = field(default_factory=lambda: np.array([]))
    setpoint: np.ndarray = field(default_factory=lambda: np.array([]))
    control_output: np.ndarray = field(default_factory=lambda: np.array([]))
    error: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Performance metrics
    iae: float = 0.0      # Integral of Absolute Error
    ise: float = 0.0      # Integral of Squared Error
    itae: float = 0.0     # Integral of Time-weighted Absolute Error
    rise_time: float = 0.0
    settling_time: float = 0.0
    overshoot: float = 0.0
    steady_state_error: float = 0.0
    
    # Controller-specific data
    controller_name: str = ""
    gains_history: Optional[Dict[str, np.ndarray]] = None


class Simulator:
    """
    Simulation manager for temperature control experiments.
    
    Handles the simulation loop, data collection, and metric computation
    for comparing different controller performances.
    """
    
    def __init__(self, system: TemperatureSystem, dt: float = 0.1):
        """
        Initialize the simulator.
        
        Args:
            system: Temperature system to control
            dt: Simulation time step (seconds)
        """
        self.system = system
        self.dt = dt
    
    def run(self, controller: Any, setpoint_func, 
            duration: float, controller_name: str = "") -> SimulationResult:
        """
        Run a simulation with the specified controller.
        
        Args:
            controller: Controller object with compute(setpoint, measurement) method
            setpoint_func: Function that returns setpoint for given time, or constant value
            duration: Total simulation duration (seconds)
            controller_name: Name for identifying this controller in results
            
        Returns:
            SimulationResult containing all simulation data
        """
        # Reset system and controller
        self.system.reset()
        if hasattr(controller, 'reset'):
            controller.reset()
        
        # Setup time vector
        num_steps = int(duration / self.dt)
        
        # Initialize data arrays
        time_data = np.zeros(num_steps)
        temp_data = np.zeros(num_steps)
        setpoint_data = np.zeros(num_steps)
        control_data = np.zeros(num_steps)
        error_data = np.zeros(num_steps)
        
        # Track gains for fuzzy PID
        track_gains = hasattr(controller, 'get_gains') and hasattr(controller, 'base_kp')
        if track_gains:
            kp_history = np.zeros(num_steps)
            ki_history = np.zeros(num_steps)
            kd_history = np.zeros(num_steps)
        
        # Simulation loop
        for i in range(num_steps):
            current_time = i * self.dt
            
            # Get setpoint (handle both function and constant)
            if callable(setpoint_func):
                setpoint = setpoint_func(current_time)
            else:
                setpoint = float(setpoint_func)
            
            # Get measurement
            measurement = self.system.get_measured_temperature()
            
            # Compute control action
            control = controller.compute(setpoint, measurement)
            
            # Step system
            new_temp = self.system.step(control)
            
            # Store data
            time_data[i] = current_time
            temp_data[i] = new_temp
            setpoint_data[i] = setpoint
            control_data[i] = control
            error_data[i] = setpoint - new_temp
            
            # Track gains
            if track_gains:
                gains = controller.get_gains()
                kp_history[i] = gains['Kp']
                ki_history[i] = gains['Ki']
                kd_history[i] = gains['Kd']
        
        # Create result object
        result = SimulationResult(
            time=time_data,
            temperature=temp_data,
            setpoint=setpoint_data,
            control_output=control_data,
            error=error_data,
            controller_name=controller_name
        )
        
        if track_gains:
            result.gains_history = {
                'Kp': kp_history,
                'Ki': ki_history,
                'Kd': kd_history
            }
        
        # Compute metrics
        self._compute_metrics(result)
        
        return result
    
    def _compute_metrics(self, result: SimulationResult):
        """
        Compute performance metrics from simulation data.
        
        Args:
            result: SimulationResult to populate with metrics
        """
        time = result.time
        error = result.error
        temp = result.temperature
        setpoint = result.setpoint
        
        # Integral error metrics
        # Use np.trapezoid (NumPy 2.x) with fallback to np.trapz (NumPy 1.x)
        try:
            trapz_func = np.trapezoid
        except AttributeError:
            trapz_func = np.trapz
        
        result.iae = trapz_func(np.abs(error), time)
        result.ise = trapz_func(error ** 2, time)
        result.itae = trapz_func(time * np.abs(error), time)
        
        # Get final setpoint for transient analysis
        final_setpoint = setpoint[-1]
        initial_temp = temp[0]
        
        # Rise time (10% to 90% of step change)
        if final_setpoint != initial_temp:
            step_size = final_setpoint - initial_temp
            target_10 = initial_temp + 0.1 * step_size
            target_90 = initial_temp + 0.9 * step_size
            
            if step_size > 0:
                idx_10 = np.argmax(temp >= target_10) if np.any(temp >= target_10) else -1
                idx_90 = np.argmax(temp >= target_90) if np.any(temp >= target_90) else -1
            else:
                idx_10 = np.argmax(temp <= target_10) if np.any(temp <= target_10) else -1
                idx_90 = np.argmax(temp <= target_90) if np.any(temp <= target_90) else -1
            
            if idx_10 >= 0 and idx_90 >= 0 and idx_90 > idx_10:
                result.rise_time = time[idx_90] - time[idx_10]
            else:
                result.rise_time = float('inf')
        
        # Settling time (2% band)
        tolerance = 0.02 * abs(final_setpoint) if final_setpoint != 0 else 0.5
        settled_indices = np.where(np.abs(error) <= tolerance)[0]
        
        if len(settled_indices) > 0:
            # Find first index where it stays settled
            for i in range(len(settled_indices)):
                idx = settled_indices[i]
                if idx >= len(time) - 1:
                    result.settling_time = time[idx]
                    break
                # Check if it stays within tolerance
                if np.all(np.abs(error[idx:]) <= tolerance):
                    result.settling_time = time[idx]
                    break
            else:
                result.settling_time = time[-1]
        else:
            result.settling_time = float('inf')
        
        # Overshoot percentage
        if final_setpoint > initial_temp:
            peak = np.max(temp)
            if peak > final_setpoint:
                result.overshoot = ((peak - final_setpoint) / 
                                    (final_setpoint - initial_temp)) * 100
            else:
                result.overshoot = 0.0
        elif final_setpoint < initial_temp:
            trough = np.min(temp)
            if trough < final_setpoint:
                result.overshoot = ((final_setpoint - trough) / 
                                    (initial_temp - final_setpoint)) * 100
            else:
                result.overshoot = 0.0
        else:
            result.overshoot = 0.0
        
        # Steady-state error (average of last 10% of simulation)
        last_portion = int(len(error) * 0.1)
        if last_portion > 0:
            result.steady_state_error = np.mean(np.abs(error[-last_portion:]))
        else:
            result.steady_state_error = abs(error[-1])
    
    def compare_controllers(self, controllers: List[tuple], 
                           setpoint_func, duration: float) -> Dict[str, SimulationResult]:
        """
        Run simulations for multiple controllers and compare results.
        
        Args:
            controllers: List of (name, controller) tuples
            setpoint_func: Setpoint function or constant value
            duration: Simulation duration
            
        Returns:
            Dictionary mapping controller names to their results
        """
        results = {}
        
        for name, controller in controllers:
            results[name] = self.run(controller, setpoint_func, duration, name)
        
        return results


def create_setpoint_profile(changes: List[tuple]) -> callable:
    """
    Create a setpoint function from a list of (time, value) pairs.
    
    The setpoint holds at each value until the next change time.
    
    Args:
        changes: List of (time, setpoint_value) tuples, sorted by time
        
    Returns:
        Function that returns setpoint for any given time
    """
    def setpoint_func(t: float) -> float:
        # Find the applicable setpoint
        current_setpoint = changes[0][1]
        for change_time, value in changes:
            if t >= change_time:
                current_setpoint = value
            else:
                break
        return current_setpoint
    
    return setpoint_func


def print_metrics_comparison(results: Dict[str, SimulationResult]):
    """
    Print a formatted comparison of simulation metrics.
    
    Args:
        results: Dictionary of controller name -> SimulationResult
    """
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS COMPARISON")
    print("=" * 70)
    
    # Header
    header = f"{'Metric':<25}"
    for name in results.keys():
        header += f"{name:>15}"
    print(header)
    print("-" * 70)
    
    # Metrics
    metrics = [
        ('IAE', 'iae', '.2f'),
        ('ISE', 'ise', '.2f'),
        ('ITAE', 'itae', '.2f'),
        ('Rise Time (s)', 'rise_time', '.2f'),
        ('Settling Time (s)', 'settling_time', '.2f'),
        ('Overshoot (%)', 'overshoot', '.2f'),
        ('Steady-State Error', 'steady_state_error', '.4f'),
    ]
    
    for label, attr, fmt in metrics:
        row = f"{label:<25}"
        for name, result in results.items():
            value = getattr(result, attr)
            if value == float('inf'):
                row += f"{'N/A':>15}"
            else:
                row += f"{value:>15{fmt}}"
        print(row)
    
    print("=" * 70)
