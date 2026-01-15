"""
Fuzzy Rules Module

This module implements the rule base for fuzzy inference systems.
Fuzzy rules follow the IF-THEN structure and encode expert knowledge
about how to control a system based on linguistic variables.

A typical rule has the form:
IF error is NB AND change_error is NS THEN output is PM
"""

from typing import Dict, List, Tuple, Optional
from .fuzzy_sets import MembershipFunction


class FuzzyRule:
    """
    Represents a single fuzzy IF-THEN rule.
    
    A fuzzy rule consists of:
    - Antecedent: A set of conditions (fuzzy propositions) connected by AND/OR
    - Consequent: The output fuzzy set when the rule fires
    - Weight: Optional importance factor for the rule (default 1.0)
    """
    
    def __init__(self, antecedent: Dict[str, str], consequent: str, 
                 weight: float = 1.0):
        """
        Initialize a fuzzy rule.
        
        Args:
            antecedent: Dictionary mapping input variable names to their 
                       required linguistic term (e.g., {'error': 'NB', 'de': 'PS'})
            consequent: The output linguistic term when rule fires
            weight: Importance weight of this rule (0.0 to 1.0)
        """
        self.antecedent = antecedent
        self.consequent = consequent
        self.weight = weight
    
    def evaluate(self, inputs: Dict[str, float], 
                 input_mfs: Dict[str, Dict[str, MembershipFunction]],
                 operator: str = 'min') -> float:
        """
        Evaluate the firing strength of this rule.
        
        The firing strength is computed by combining the membership degrees
        of all antecedent propositions using the specified T-norm operator.
        
        Args:
            inputs: Dictionary of crisp input values
            input_mfs: Nested dictionary of membership functions for each input
            operator: Fuzzy AND operator ('min' for minimum, 'prod' for product)
            
        Returns:
            Firing strength of the rule (0.0 to 1.0)
        """
        membership_degrees = []
        
        for var_name, term_name in self.antecedent.items():
            if var_name not in inputs:
                raise ValueError(f"Input variable '{var_name}' not provided")
            if var_name not in input_mfs:
                raise ValueError(f"No membership functions for '{var_name}'")
            if term_name not in input_mfs[var_name]:
                raise ValueError(f"Unknown term '{term_name}' for '{var_name}'")
            
            crisp_value = inputs[var_name]
            mf = input_mfs[var_name][term_name]
            degree = mf.compute(crisp_value)
            membership_degrees.append(degree)
        
        # Apply T-norm (fuzzy AND)
        if operator == 'min':
            firing_strength = min(membership_degrees) if membership_degrees else 0.0
        elif operator == 'prod':
            firing_strength = 1.0
            for d in membership_degrees:
                firing_strength *= d
        else:
            raise ValueError(f"Unknown operator: {operator}")
        
        return firing_strength * self.weight
    
    def __repr__(self) -> str:
        """String representation of the rule."""
        antecedent_str = " AND ".join([f"{k} is {v}" 
                                       for k, v in self.antecedent.items()])
        return f"IF {antecedent_str} THEN output is {self.consequent}"


class RuleBase:
    """
    A collection of fuzzy rules forming the knowledge base.
    
    The rule base encodes the expert knowledge used for fuzzy inference.
    Rules can be added individually or generated automatically for common
    control scenarios.
    """
    
    def __init__(self):
        """Initialize an empty rule base."""
        self.rules: List[FuzzyRule] = []
    
    def add_rule(self, rule: FuzzyRule):
        """
        Add a single rule to the rule base.
        
        Args:
            rule: FuzzyRule to add
        """
        self.rules.append(rule)
    
    def add_rules(self, rules: List[FuzzyRule]):
        """
        Add multiple rules to the rule base.
        
        Args:
            rules: List of FuzzyRule objects to add
        """
        self.rules.extend(rules)
    
    def get_rules(self) -> List[FuzzyRule]:
        """Get all rules in the rule base."""
        return self.rules
    
    def __len__(self) -> int:
        """Return the number of rules."""
        return len(self.rules)
    
    def __iter__(self):
        """Iterate over rules."""
        return iter(self.rules)


def create_pid_rule_base() -> Tuple[RuleBase, RuleBase, RuleBase]:
    """
    Create the fuzzy rule bases for a Fuzzy PID controller.
    
    This function generates the standard 25-rule (5x5) matrix for adjusting
    the P, I, and D gains based on error (e) and change in error (de).
    
    The rules are designed according to established control theory:
    - When error is large, use aggressive control action
    - When error is small and stable, use minimal control action
    - Derivative action helps anticipate future error trends
    
    Returns:
        Tuple of (Kp_rules, Ki_rules, Kd_rules)
    """
    # Linguistic terms: NB=Negative Big, NS=Negative Small, ZE=Zero,
    #                   PS=Positive Small, PB=Positive Big
    terms = ['NB', 'NS', 'ZE', 'PS', 'PB']
    
    # Rule tables adapted from fuzzy PID control literature
    # Each table shows the output term for given (error, delta_error) combination
    
    # Kp adjustment rules - Proportional gain
    # High Kp when error is large, reduce as we approach setpoint
    kp_table = [
        # de:  NB    NS    ZE    PS    PB   <- Change in error
        ['PB', 'PB', 'PM', 'PM', 'PS'],  # NB  (e)
        ['PB', 'PM', 'PM', 'PS', 'ZE'],  # NS
        ['PM', 'PM', 'PS', 'ZE', 'NS'],  # ZE
        ['PM', 'PS', 'ZE', 'NS', 'NM'],  # PS
        ['PS', 'ZE', 'NS', 'NM', 'NB'],  # PB
    ]
    
    # Ki adjustment rules - Integral gain
    # Low Ki when error is large (prevent windup), increase when stable
    ki_table = [
        # de:  NB    NS    ZE    PS    PB
        ['NB', 'NB', 'NM', 'NS', 'ZE'],  # NB
        ['NB', 'NM', 'NS', 'ZE', 'PS'],  # NS
        ['NM', 'NS', 'ZE', 'PS', 'PM'],  # ZE
        ['NS', 'ZE', 'PS', 'PM', 'PB'],  # PS
        ['ZE', 'PS', 'PM', 'PB', 'PB'],  # PB
    ]
    
    # Kd adjustment rules - Derivative gain
    # High Kd when change is large, reduce for stability
    kd_table = [
        # de:  NB    NS    ZE    PS    PB
        ['PS', 'PS', 'ZE', 'NS', 'NB'],  # NB
        ['PB', 'PS', 'ZE', 'NS', 'NM'],  # NS
        ['PB', 'PM', 'ZE', 'NM', 'NB'],  # ZE
        ['PM', 'PS', 'ZE', 'NS', 'NB'],  # PS
        ['PS', 'ZE', 'NS', 'NM', 'NB'],  # PB
    ]
    
    # Create rule bases with 7-term output (for finer adjustment)
    output_terms = ['NB', 'NM', 'NS', 'ZE', 'PS', 'PM', 'PB']
    
    # Remap 5-term outputs to work with our 7-term tables
    kp_rules = RuleBase()
    ki_rules = RuleBase()
    kd_rules = RuleBase()
    
    for i, e_term in enumerate(terms):
        for j, de_term in enumerate(terms):
            # Create antecedent for all rules
            antecedent = {'error': e_term, 'delta_error': de_term}
            
            # Add Kp rule
            kp_rules.add_rule(FuzzyRule(antecedent, kp_table[i][j]))
            
            # Add Ki rule
            ki_rules.add_rule(FuzzyRule(antecedent, ki_table[i][j]))
            
            # Add Kd rule
            kd_rules.add_rule(FuzzyRule(antecedent, kd_table[i][j]))
    
    return kp_rules, ki_rules, kd_rules


def create_simple_rule_base() -> RuleBase:
    """
    Create a simplified rule base for direct fuzzy control output.
    
    This 25-rule base maps error and delta_error directly to control action,
    suitable for simple fuzzy controllers without PID structure.
    
    Returns:
        RuleBase with 25 rules for direct control
    """
    terms = ['NB', 'NS', 'ZE', 'PS', 'PB']
    
    # Direct control output table
    # When error is negative (below setpoint), increase output
    # When error is positive (above setpoint), decrease output
    output_table = [
        # de:  NB    NS    ZE    PS    PB
        ['PB', 'PB', 'PM', 'PS', 'ZE'],  # NB (e) - way below setpoint
        ['PB', 'PM', 'PS', 'ZE', 'NS'],  # NS
        ['PM', 'PS', 'ZE', 'NS', 'NM'],  # ZE
        ['PS', 'ZE', 'NS', 'NM', 'NB'],  # PS
        ['ZE', 'NS', 'NM', 'NB', 'NB'],  # PB - way above setpoint
    ]
    
    rule_base = RuleBase()
    
    for i, e_term in enumerate(terms):
        for j, de_term in enumerate(terms):
            antecedent = {'error': e_term, 'delta_error': de_term}
            rule_base.add_rule(FuzzyRule(antecedent, output_table[i][j]))
    
    return rule_base
