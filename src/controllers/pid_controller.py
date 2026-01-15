"""
Conventional PID Controller Module

This module implements a standard Proportional-Integral-Derivative (PID)
controller that serves as the baseline for comparison with the Fuzzy PID
controller.

The PID control law is:
    u(t) = Kp * e(t) + Ki * integral(e(t)) + Kd * d(e(t))/dt

where:
    - e(t) = setpoint - measured_value (error)
    - Kp = Proportional gain
    - Ki = Integral gain
    - Kd = Derivative gain
"""


class PIDController:
    """
    Discrete-time PID Controller implementation.
    
    Features:
    - Anti-windup mechanism to prevent integral saturation
    - Output saturation limits
    - Derivative filtering to reduce noise amplification
    """
    
    def __init__(self, kp: float, ki: float, kd: float,
                 dt: float = 0.1,
                 output_min: float = 0.0, 
                 output_max: float = 100.0,
                 anti_windup: bool = True):
        """
        Initialize the PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            dt: Sampling time (seconds)
            output_min: Minimum output value (saturation limit)
            output_max: Maximum output value (saturation limit)
            anti_windup: Enable anti-windup mechanism
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_min = output_min
        self.output_max = output_max
        self.anti_windup = anti_windup
        
        # Controller state
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_derivative = 0.0
        
        # Derivative filter coefficient (low-pass filter)
        self.derivative_filter_coeff = 0.1
        
    def reset(self):
        """
        Reset the controller state.
        
        Call this when starting a new control episode or when
        the setpoint changes significantly.
        """
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_derivative = 0.0
    
    def compute(self, setpoint: float, measured_value: float) -> float:
        """
        Compute the control output for given setpoint and measurement.
        
        Args:
            setpoint: Desired value
            measured_value: Current measured value
            
        Returns:
            Control output (saturated to [output_min, output_max])
        """
        # Calculate error
        error = setpoint - measured_value
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        potential_integral = self.integral + error * self.dt
        i_term = self.ki * potential_integral
        
        # Derivative term with filtering
        raw_derivative = (error - self.previous_error) / self.dt
        filtered_derivative = (self.derivative_filter_coeff * raw_derivative + 
                              (1 - self.derivative_filter_coeff) * self.previous_derivative)
        d_term = self.kd * filtered_derivative
        
        # Compute raw output
        output = p_term + i_term + d_term
        
        # Apply saturation
        saturated_output = max(self.output_min, min(self.output_max, output))
        
        # Anti-windup: only integrate if output is not saturated
        if self.anti_windup:
            if saturated_output == output:
                # Not saturated, update integral normally
                self.integral = potential_integral
            # If saturated, don't update integral (prevent windup)
        else:
            self.integral = potential_integral
        
        # Update state for next iteration
        self.previous_error = error
        self.previous_derivative = filtered_derivative
        
        return saturated_output
    
    def get_terms(self, setpoint: float, measured_value: float) -> dict:
        """
        Get individual PID terms for analysis.
        
        Args:
            setpoint: Desired value
            measured_value: Current measured value
            
        Returns:
            Dictionary with P, I, D terms and total output
        """
        error = setpoint - measured_value
        
        p_term = self.kp * error
        i_term = self.ki * self.integral
        
        raw_derivative = (error - self.previous_error) / self.dt
        d_term = self.kd * raw_derivative
        
        return {
            'error': error,
            'P': p_term,
            'I': i_term,
            'D': d_term,
            'total': p_term + i_term + d_term
        }
    
    def set_gains(self, kp: float = None, ki: float = None, kd: float = None):
        """
        Update controller gains.
        
        Args:
            kp: New proportional gain (or None to keep current)
            ki: New integral gain (or None to keep current)
            kd: New derivative gain (or None to keep current)
        """
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd
    
    def get_gains(self) -> dict:
        """
        Get current controller gains.
        
        Returns:
            Dictionary with Kp, Ki, Kd values
        """
        return {'Kp': self.kp, 'Ki': self.ki, 'Kd': self.kd}
    
    def __repr__(self) -> str:
        """String representation of the controller."""
        return f"PIDController(Kp={self.kp}, Ki={self.ki}, Kd={self.kd})"
