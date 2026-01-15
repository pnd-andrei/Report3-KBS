"""
Temperature System Simulation Module

This module implements a first-order thermal system model for simulating
temperature control scenarios. The model represents a simplified but
realistic thermal process with configurable parameters.

The system dynamics are described by:
    dT/dt = (1/tau) * (K * u(t) - (T - T_ambient)) + d(t)

where:
    - T: System temperature
    - tau: Time constant (thermal inertia)
    - K: Process gain (heating power to temperature relationship)
    - u(t): Control input (heating power, 0-100%)
    - T_ambient: Ambient temperature
    - d(t): External disturbance
"""

import numpy as np
from typing import Optional, Callable


class TemperatureSystem:
    """
    First-order thermal system model.
    
    Simulates the temperature dynamics of a thermal process such as
    a heating chamber, HVAC system, or industrial furnace.
    
    Features:
    - Configurable thermal parameters
    - External disturbance injection
    - Measurement noise simulation
    - Dead time (transport delay) support
    """
    
    def __init__(self,
                 time_constant: float = 50.0,
                 gain: float = 1.0,
                 ambient_temp: float = 20.0,
                 initial_temp: float = 25.0,
                 dead_time: float = 0.0,
                 dt: float = 0.1,
                 noise_std: float = 0.0):
        """
        Initialize the temperature system.
        
        Args:
            time_constant: Thermal time constant (seconds)
            gain: Process gain (degrees per 1% control input)
            ambient_temp: Ambient/room temperature (degrees C)
            initial_temp: Initial system temperature (degrees C)
            dead_time: Transport delay (seconds)
            dt: Simulation time step (seconds)
            noise_std: Standard deviation of measurement noise
        """
        self.tau = time_constant
        self.gain = gain
        self.T_ambient = ambient_temp
        self.initial_temp = initial_temp
        self.dead_time = dead_time
        self.dt = dt
        self.noise_std = noise_std
        
        # Current state
        self.temperature = initial_temp
        self.time = 0.0
        
        # Dead time buffer
        self.dead_time_steps = int(dead_time / dt) if dead_time > 0 else 0
        self.control_buffer = [0.0] * max(1, self.dead_time_steps)
        
        # Disturbance function
        self.disturbance_func: Optional[Callable[[float], float]] = None
    
    def reset(self, initial_temp: Optional[float] = None):
        """
        Reset the system to initial conditions.
        
        Args:
            initial_temp: Optional new initial temperature
        """
        if initial_temp is not None:
            self.initial_temp = initial_temp
        self.temperature = self.initial_temp
        self.time = 0.0
        self.control_buffer = [0.0] * max(1, self.dead_time_steps)
    
    def set_disturbance(self, disturbance_func: Callable[[float], float]):
        """
        Set a disturbance function.
        
        The disturbance function takes time as input and returns the
        disturbance value to add to the temperature derivative.
        
        Args:
            disturbance_func: Function(time) -> disturbance value
        """
        self.disturbance_func = disturbance_func
    
    def clear_disturbance(self):
        """Remove any set disturbance function."""
        self.disturbance_func = None
    
    def step(self, control_input: float) -> float:
        """
        Advance the simulation by one time step.
        
        Args:
            control_input: Heating power (0-100%)
            
        Returns:
            Current measured temperature (with noise if configured)
        """
        # Saturate control input
        control = max(0.0, min(100.0, control_input))
        
        # Handle dead time
        if self.dead_time_steps > 0:
            effective_control = self.control_buffer[0]
            self.control_buffer.pop(0)
            self.control_buffer.append(control)
        else:
            effective_control = control
        
        # Get disturbance
        disturbance = 0.0
        if self.disturbance_func is not None:
            disturbance = self.disturbance_func(self.time)
        
        # First-order dynamics: dT/dt = (1/tau) * (K*u - (T - T_ambient)) + d
        # Using Euler integration
        heating_effect = self.gain * effective_control
        cooling_effect = self.temperature - self.T_ambient
        
        dT_dt = (1.0 / self.tau) * (heating_effect - cooling_effect) + disturbance
        
        self.temperature += dT_dt * self.dt
        self.time += self.dt
        
        # Return measured temperature with noise
        measurement = self.temperature
        if self.noise_std > 0:
            measurement += np.random.normal(0, self.noise_std)
        
        return measurement
    
    def get_temperature(self) -> float:
        """Get current temperature (without noise)."""
        return self.temperature
    
    def get_measured_temperature(self) -> float:
        """Get current temperature with measurement noise."""
        if self.noise_std > 0:
            return self.temperature + np.random.normal(0, self.noise_std)
        return self.temperature
    
    def get_time(self) -> float:
        """Get current simulation time."""
        return self.time
    
    def get_steady_state_temperature(self, control_input: float) -> float:
        """
        Calculate the steady-state temperature for a given control input.
        
        At steady state, dT/dt = 0, so:
        K * u = T_ss - T_ambient
        T_ss = K * u + T_ambient
        
        Args:
            control_input: Heating power (0-100%)
            
        Returns:
            Steady-state temperature
        """
        return self.gain * control_input + self.T_ambient
    
    def get_required_control(self, target_temp: float) -> float:
        """
        Calculate the control input required to reach a target temperature.
        
        Args:
            target_temp: Desired steady-state temperature
            
        Returns:
            Required control input (0-100%)
        """
        control = (target_temp - self.T_ambient) / self.gain
        return max(0.0, min(100.0, control))
    
    def __repr__(self) -> str:
        """String representation of the system."""
        return (f"TemperatureSystem(tau={self.tau}, gain={self.gain}, "
                f"T_ambient={self.T_ambient})")


# Predefined disturbance functions
def step_disturbance(time: float, start_time: float = 50.0, 
                     magnitude: float = 5.0) -> float:
    """
    Step disturbance that activates at a specified time.
    
    Args:
        time: Current simulation time
        start_time: Time when disturbance starts
        magnitude: Size of the disturbance
        
    Returns:
        Disturbance value
    """
    return magnitude if time >= start_time else 0.0


def pulse_disturbance(time: float, start_time: float = 50.0,
                      duration: float = 10.0, magnitude: float = 10.0) -> float:
    """
    Pulse disturbance lasting for a specified duration.
    
    Args:
        time: Current simulation time
        start_time: Time when pulse starts
        duration: How long the pulse lasts
        magnitude: Size of the disturbance
        
    Returns:
        Disturbance value
    """
    if start_time <= time < start_time + duration:
        return magnitude
    return 0.0


def sinusoidal_disturbance(time: float, amplitude: float = 2.0,
                           frequency: float = 0.1, phase: float = 0.0) -> float:
    """
    Sinusoidal disturbance for testing oscillation handling.
    
    Args:
        time: Current simulation time
        amplitude: Peak amplitude
        frequency: Oscillation frequency (Hz)
        phase: Phase offset (radians)
        
    Returns:
        Disturbance value
    """
    return amplitude * np.sin(2 * np.pi * frequency * time + phase)
