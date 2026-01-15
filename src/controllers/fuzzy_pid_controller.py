"""
Fuzzy PID Controller Module

This module implements a Fuzzy PID Controller that uses fuzzy logic to
dynamically adjust PID gains based on the current error and rate of change
of error. This approach combines the advantages of both fuzzy logic
(handling nonlinearity and uncertainty) and PID control (proven stability).

The controller uses a 25-rule Mamdani fuzzy inference system to compute
adjustment factors for Kp, Ki, and Kd based on:
- Error (e): difference between setpoint and measured value
- Delta Error (de): rate of change of error

Architecture:
    [Error, Delta Error] -> Fuzzy System -> [Kp_adj, Ki_adj, Kd_adj] -> PID
"""

import numpy as np
from typing import Dict, Tuple
from ..fuzzy_logic.fuzzy_sets import (
    MembershipFunction, TriangularMF, GaussianMF, 
    create_linguistic_variable
)
from ..fuzzy_logic.fuzzy_rules import RuleBase, FuzzyRule
from ..fuzzy_logic.inference_engine import FuzzyInferenceEngine


class FuzzyPIDController:
    """
    Fuzzy PID Controller with self-tuning capabilities.
    
    This controller dynamically adjusts PID gains using fuzzy logic,
    providing improved performance over conventional fixed-gain PID,
    especially for nonlinear systems and changing operating conditions.
    """
    
    def __init__(self, 
                 base_kp: float = 2.0, 
                 base_ki: float = 0.5, 
                 base_kd: float = 0.1,
                 dt: float = 0.1,
                 error_range: Tuple[float, float] = (-50, 50),
                 delta_error_range: Tuple[float, float] = (-20, 20),
                 output_min: float = 0.0,
                 output_max: float = 100.0,
                 mf_type: str = 'triangular'):
        """
        Initialize the Fuzzy PID controller.
        
        Args:
            base_kp: Base proportional gain
            base_ki: Base integral gain  
            base_kd: Base derivative gain
            dt: Sampling time (seconds)
            error_range: Expected range of error values
            delta_error_range: Expected range of error change values
            output_min: Minimum controller output
            output_max: Maximum controller output
            mf_type: Membership function type ('triangular' or 'gaussian')
        """
        self.base_kp = base_kp
        self.base_ki = base_ki
        self.base_kd = base_kd
        self.dt = dt
        self.output_min = output_min
        self.output_max = output_max
        self.mf_type = mf_type
        
        # Current effective gains
        self.kp = base_kp
        self.ki = base_ki
        self.kd = base_kd
        
        # Controller state
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_derivative = 0.0
        
        # Derivative filter coefficient
        self.derivative_filter_coeff = 0.1
        
        # Setup fuzzy inference engines for each gain adjustment
        self._setup_fuzzy_engines(error_range, delta_error_range, mf_type)
    
    def _setup_fuzzy_engines(self, error_range: Tuple[float, float],
                             delta_error_range: Tuple[float, float],
                             mf_type: str):
        """
        Setup the fuzzy inference engines for gain adjustment.
        
        Creates three inference engines (one each for Kp, Ki, Kd)
        with appropriate membership functions and rule bases.
        """
        # Define input membership functions
        self.input_mfs = {
            'error': create_linguistic_variable(
                'error', error_range[0], error_range[1], 
                num_terms=5, mf_type=mf_type
            ),
            'delta_error': create_linguistic_variable(
                'delta_error', delta_error_range[0], delta_error_range[1],
                num_terms=5, mf_type=mf_type
            )
        }
        
        # Define output membership functions for gain adjustments
        # Output range: -1 to 1 (multiplier adjustment factor)
        output_range = (-1.0, 1.0)
        self.output_mfs = create_linguistic_variable(
            'adjustment', output_range[0], output_range[1],
            num_terms=7, mf_type=mf_type
        )
        
        # Create rule bases for each PID gain
        self.kp_rules = self._create_kp_rule_base()
        self.ki_rules = self._create_ki_rule_base()
        self.kd_rules = self._create_kd_rule_base()
        
        # Create inference engines
        self.kp_engine = FuzzyInferenceEngine(
            self.input_mfs, self.output_mfs, self.kp_rules,
            output_range, resolution=100
        )
        self.ki_engine = FuzzyInferenceEngine(
            self.input_mfs, self.output_mfs, self.ki_rules,
            output_range, resolution=100
        )
        self.kd_engine = FuzzyInferenceEngine(
            self.input_mfs, self.output_mfs, self.kd_rules,
            output_range, resolution=100
        )
    
    def _create_kp_rule_base(self) -> RuleBase:
        """
        Create the rule base for Kp adjustment.
        
        Strategy: High Kp when error is large to drive system quickly,
        reduce Kp as we approach setpoint to prevent overshoot.
        """
        terms = ['NB', 'NS', 'ZE', 'PS', 'PB']
        output_terms = ['NB', 'NM', 'NS', 'ZE', 'PS', 'PM', 'PB']
        
        # Kp adjustment table (absolute error magnitude matters)
        kp_table = [
            ['PB', 'PB', 'PM', 'PM', 'PS'],  # NB error
            ['PB', 'PM', 'PM', 'PS', 'ZE'],  # NS error
            ['PM', 'PM', 'PS', 'ZE', 'NS'],  # ZE error
            ['PM', 'PS', 'ZE', 'NS', 'NM'],  # PS error
            ['PS', 'ZE', 'NS', 'NM', 'NB'],  # PB error
        ]
        
        rule_base = RuleBase()
        for i, e_term in enumerate(terms):
            for j, de_term in enumerate(terms):
                rule_base.add_rule(FuzzyRule(
                    {'error': e_term, 'delta_error': de_term},
                    kp_table[i][j]
                ))
        
        return rule_base
    
    def _create_ki_rule_base(self) -> RuleBase:
        """
        Create the rule base for Ki adjustment.
        
        Strategy: Low Ki when error is large (prevent windup),
        increase Ki when error is small and stable to eliminate steady-state error.
        """
        terms = ['NB', 'NS', 'ZE', 'PS', 'PB']
        
        # Ki adjustment table
        ki_table = [
            ['NB', 'NB', 'NM', 'NS', 'ZE'],  # NB error
            ['NB', 'NM', 'NS', 'ZE', 'PS'],  # NS error
            ['NM', 'NS', 'ZE', 'PS', 'PM'],  # ZE error
            ['NS', 'ZE', 'PS', 'PM', 'PB'],  # PS error
            ['ZE', 'PS', 'PM', 'PB', 'PB'],  # PB error
        ]
        
        rule_base = RuleBase()
        for i, e_term in enumerate(terms):
            for j, de_term in enumerate(terms):
                rule_base.add_rule(FuzzyRule(
                    {'error': e_term, 'delta_error': de_term},
                    ki_table[i][j]
                ))
        
        return rule_base
    
    def _create_kd_rule_base(self) -> RuleBase:
        """
        Create the rule base for Kd adjustment.
        
        Strategy: High Kd when change in error is large (anticipate changes),
        reduce Kd when system is stable to prevent noise amplification.
        """
        terms = ['NB', 'NS', 'ZE', 'PS', 'PB']
        
        # Kd adjustment table
        kd_table = [
            ['PS', 'PS', 'ZE', 'NS', 'NB'],  # NB error
            ['PB', 'PS', 'ZE', 'NS', 'NM'],  # NS error
            ['PB', 'PM', 'ZE', 'NM', 'NB'],  # ZE error
            ['PM', 'PS', 'ZE', 'NS', 'NB'],  # PS error
            ['PS', 'ZE', 'NS', 'NM', 'NB'],  # PB error
        ]
        
        rule_base = RuleBase()
        for i, e_term in enumerate(terms):
            for j, de_term in enumerate(terms):
                rule_base.add_rule(FuzzyRule(
                    {'error': e_term, 'delta_error': de_term},
                    kd_table[i][j]
                ))
        
        return rule_base
    
    def reset(self):
        """Reset the controller state."""
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_derivative = 0.0
        self.kp = self.base_kp
        self.ki = self.base_ki
        self.kd = self.base_kd
    
    def _update_gains(self, error: float, delta_error: float):
        """
        Update PID gains using fuzzy inference.
        
        Args:
            error: Current error value
            delta_error: Current rate of change of error
        """
        inputs = {'error': error, 'delta_error': delta_error}
        
        # Get adjustment factors from fuzzy inference
        kp_adj = self.kp_engine.infer(inputs, defuzz_method='cog')
        ki_adj = self.ki_engine.infer(inputs, defuzz_method='cog')
        kd_adj = self.kd_engine.infer(inputs, defuzz_method='cog')
        
        # Apply adjustments to base gains
        # Scale factors: adjustment range [-1, 1] maps to gain multiplier [0.5, 1.5]
        self.kp = self.base_kp * (1.0 + 0.5 * kp_adj)
        self.ki = self.base_ki * (1.0 + 0.5 * ki_adj)
        self.kd = self.base_kd * (1.0 + 0.5 * kd_adj)
        
        # Ensure gains remain positive
        self.kp = max(0.01, self.kp)
        self.ki = max(0.0, self.ki)
        self.kd = max(0.0, self.kd)
    
    def compute(self, setpoint: float, measured_value: float) -> float:
        """
        Compute the control output.
        
        Args:
            setpoint: Desired value
            measured_value: Current measured value
            
        Returns:
            Control output (saturated to [output_min, output_max])
        """
        # Calculate error
        error = setpoint - measured_value
        
        # Calculate delta error
        delta_error = (error - self.previous_error) / self.dt
        
        # Update gains using fuzzy logic
        self._update_gains(error, delta_error)
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        potential_integral = self.integral + error * self.dt
        i_term = self.ki * potential_integral
        
        # Derivative term with filtering
        filtered_derivative = (self.derivative_filter_coeff * delta_error + 
                              (1 - self.derivative_filter_coeff) * self.previous_derivative)
        d_term = self.kd * filtered_derivative
        
        # Compute raw output
        output = p_term + i_term + d_term
        
        # Apply saturation
        saturated_output = max(self.output_min, min(self.output_max, output))
        
        # Anti-windup
        if saturated_output == output:
            self.integral = potential_integral
        
        # Update state for next iteration
        self.previous_error = error
        self.previous_derivative = filtered_derivative
        
        return saturated_output
    
    def get_gains(self) -> Dict[str, float]:
        """
        Get current effective controller gains.
        
        Returns:
            Dictionary with current Kp, Ki, Kd values
        """
        return {'Kp': self.kp, 'Ki': self.ki, 'Kd': self.kd}
    
    def get_base_gains(self) -> Dict[str, float]:
        """
        Get base controller gains.
        
        Returns:
            Dictionary with base Kp, Ki, Kd values
        """
        return {'Kp': self.base_kp, 'Ki': self.base_ki, 'Kd': self.base_kd}
    
    def get_terms(self, setpoint: float, measured_value: float) -> dict:
        """
        Get individual PID terms for analysis.
        
        Args:
            setpoint: Desired value
            measured_value: Current measured value
            
        Returns:
            Dictionary with P, I, D terms and gains
        """
        error = setpoint - measured_value
        delta_error = (error - self.previous_error) / self.dt
        
        p_term = self.kp * error
        i_term = self.ki * self.integral
        d_term = self.kd * self.previous_derivative
        
        return {
            'error': error,
            'delta_error': delta_error,
            'P': p_term,
            'I': i_term,
            'D': d_term,
            'Kp': self.kp,
            'Ki': self.ki,
            'Kd': self.kd,
            'total': p_term + i_term + d_term
        }
    
    def __repr__(self) -> str:
        """String representation of the controller."""
        return (f"FuzzyPIDController(base_Kp={self.base_kp}, "
                f"base_Ki={self.base_ki}, base_Kd={self.base_kd})")
