"""
Fuzzy Inference Engine Module

This module implements the Mamdani fuzzy inference system, which is the
most widely used method for fuzzy control applications due to its 
intuitive nature and interpretability.

The inference process consists of:
1. Fuzzification: Convert crisp inputs to fuzzy membership degrees
2. Rule Evaluation: Compute firing strength of each rule
3. Aggregation: Combine rule outputs
4. Defuzzification: Convert fuzzy output to crisp value
"""

import numpy as np
from typing import Dict, List, Tuple
from .fuzzy_sets import MembershipFunction, TriangularMF, GaussianMF
from .fuzzy_rules import RuleBase, FuzzyRule


class FuzzyInferenceEngine:
    """
    Mamdani-type Fuzzy Inference Engine.
    
    Implements the complete fuzzy inference process from crisp inputs
    to crisp output using the Mamdani model with Center of Gravity (COG)
    defuzzification.
    """
    
    def __init__(self, 
                 input_mfs: Dict[str, Dict[str, MembershipFunction]],
                 output_mfs: Dict[str, MembershipFunction],
                 rule_base: RuleBase,
                 output_range: Tuple[float, float],
                 resolution: int = 100):
        """
        Initialize the fuzzy inference engine.
        
        Args:
            input_mfs: Dictionary mapping input variable names to their
                      membership function dictionaries
            output_mfs: Dictionary mapping output term names to membership functions
            rule_base: The rule base containing all fuzzy rules
            output_range: Tuple of (min, max) for the output universe of discourse
            resolution: Number of points for discretizing the output space
        """
        self.input_mfs = input_mfs
        self.output_mfs = output_mfs
        self.rule_base = rule_base
        self.output_range = output_range
        self.resolution = resolution
        
        # Pre-compute output universe
        self.output_universe = np.linspace(
            output_range[0], output_range[1], resolution
        )
    
    def fuzzify(self, var_name: str, crisp_value: float) -> Dict[str, float]:
        """
        Fuzzify a crisp input value.
        
        Computes the membership degree of the input value in each fuzzy set
        defined for the variable.
        
        Args:
            var_name: Name of the input variable
            crisp_value: The crisp input value
            
        Returns:
            Dictionary mapping term names to membership degrees
        """
        if var_name not in self.input_mfs:
            raise ValueError(f"Unknown input variable: {var_name}")
        
        memberships = {}
        for term_name, mf in self.input_mfs[var_name].items():
            memberships[term_name] = mf.compute(crisp_value)
        
        return memberships
    
    def evaluate_rules(self, inputs: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        Evaluate all rules and return their outputs with firing strengths.
        
        Args:
            inputs: Dictionary of crisp input values
            
        Returns:
            List of (consequent_term, firing_strength) tuples
        """
        rule_outputs = []
        
        for rule in self.rule_base:
            firing_strength = rule.evaluate(inputs, self.input_mfs)
            if firing_strength > 0:
                rule_outputs.append((rule.consequent, firing_strength))
        
        return rule_outputs
    
    def aggregate(self, rule_outputs: List[Tuple[str, float]]) -> np.ndarray:
        """
        Aggregate rule outputs into a combined fuzzy output set.
        
        Uses the maximum operator for aggregation (union of rule outputs).
        Each rule's output is its consequent membership function clipped
        at the rule's firing strength.
        
        Args:
            rule_outputs: List of (consequent_term, firing_strength) tuples
            
        Returns:
            Aggregated membership function as an array
        """
        aggregated = np.zeros(self.resolution)
        
        for term_name, firing_strength in rule_outputs:
            if term_name not in self.output_mfs:
                # Handle cases where rule uses a term not in output MFs
                # (e.g., PM, NM in 5-term output set)
                continue
                
            mf = self.output_mfs[term_name]
            
            # Compute the output membership function
            output_mf = np.array([mf.compute(x) for x in self.output_universe])
            
            # Clip at firing strength (implication)
            clipped = np.minimum(output_mf, firing_strength)
            
            # Aggregate using maximum (union)
            aggregated = np.maximum(aggregated, clipped)
        
        return aggregated
    
    def defuzzify_cog(self, aggregated: np.ndarray) -> float:
        """
        Defuzzify using Center of Gravity (COG) method.
        
        Also known as Centroid method, this computes the center of mass
        of the aggregated fuzzy set. It is the most commonly used
        defuzzification method due to its smooth behavior.
        
        Args:
            aggregated: Aggregated membership function array
            
        Returns:
            Crisp output value
        """
        area = np.sum(aggregated)
        if area == 0:
            # Return middle of range if no rules fired
            return (self.output_range[0] + self.output_range[1]) / 2
        
        centroid = np.sum(self.output_universe * aggregated) / area
        return centroid
    
    def defuzzify_mom(self, aggregated: np.ndarray) -> float:
        """
        Defuzzify using Mean of Maxima (MOM) method.
        
        Returns the average of all values with maximum membership degree.
        Useful when a more aggressive response is desired.
        
        Args:
            aggregated: Aggregated membership function array
            
        Returns:
            Crisp output value
        """
        max_membership = np.max(aggregated)
        if max_membership == 0:
            return (self.output_range[0] + self.output_range[1]) / 2
        
        max_indices = np.where(aggregated == max_membership)[0]
        return np.mean(self.output_universe[max_indices])
    
    def defuzzify_weighted_average(self, 
                                   rule_outputs: List[Tuple[str, float]]) -> float:
        """
        Defuzzify using Weighted Average method.
        
        This is computationally efficient and similar to Sugeno defuzzification.
        Uses the peak of each output membership function weighted by
        the rule firing strength.
        
        Args:
            rule_outputs: List of (consequent_term, firing_strength) tuples
            
        Returns:
            Crisp output value
        """
        weighted_sum = 0.0
        weight_total = 0.0
        
        for term_name, firing_strength in rule_outputs:
            if term_name not in self.output_mfs:
                continue
                
            mf = self.output_mfs[term_name]
            
            # Find the peak of the membership function
            if isinstance(mf, TriangularMF):
                peak = mf.b
            elif isinstance(mf, GaussianMF):
                peak = mf.mean
            else:
                # For other MF types, find peak numerically
                peaks = np.array([mf.compute(x) for x in self.output_universe])
                peak_idx = np.argmax(peaks)
                peak = self.output_universe[peak_idx]
            
            weighted_sum += peak * firing_strength
            weight_total += firing_strength
        
        if weight_total == 0:
            return (self.output_range[0] + self.output_range[1]) / 2
        
        return weighted_sum / weight_total
    
    def infer(self, inputs: Dict[str, float], 
              defuzz_method: str = 'cog') -> float:
        """
        Perform complete fuzzy inference.
        
        This is the main entry point that performs the entire inference
        process: fuzzification, rule evaluation, aggregation, and
        defuzzification.
        
        Args:
            inputs: Dictionary mapping input variable names to crisp values
            defuzz_method: Defuzzification method ('cog', 'mom', or 'wa')
            
        Returns:
            Crisp output value
        """
        # Evaluate rules and get outputs
        rule_outputs = self.evaluate_rules(inputs)
        
        if not rule_outputs:
            # No rules fired, return neutral output
            return (self.output_range[0] + self.output_range[1]) / 2
        
        # Defuzzify
        if defuzz_method == 'wa':
            # Weighted average doesn't need aggregation
            return self.defuzzify_weighted_average(rule_outputs)
        else:
            # Aggregate for COG and MOM
            aggregated = self.aggregate(rule_outputs)
            
            if defuzz_method == 'cog':
                return self.defuzzify_cog(aggregated)
            elif defuzz_method == 'mom':
                return self.defuzzify_mom(aggregated)
            else:
                raise ValueError(f"Unknown defuzzification method: {defuzz_method}")
    
    def get_rule_activations(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """
        Get the firing strength of each rule for given inputs.
        
        Useful for debugging and understanding controller behavior.
        
        Args:
            inputs: Dictionary of crisp input values
            
        Returns:
            Dictionary mapping rule descriptions to firing strengths
        """
        activations = {}
        for rule in self.rule_base:
            firing_strength = rule.evaluate(inputs, self.input_mfs)
            activations[str(rule)] = firing_strength
        return activations
