"""
Fuzzy Sets and Membership Functions Module

This module implements various membership function types used in fuzzy logic systems.
Membership functions map crisp input values to fuzzy membership degrees in the range [0, 1].

Supported membership function types:
- Triangular (TriangularMF)
- Trapezoidal (TrapezoidalMF)  
- Gaussian (GaussianMF)
"""

import numpy as np
from abc import ABC, abstractmethod


class MembershipFunction(ABC):
    """
    Abstract base class for membership functions.
    
    A membership function defines how each point in the input space is mapped
    to a membership degree between 0 and 1. This is the fundamental building
    block of fuzzy sets.
    """
    
    def __init__(self, name: str):
        """
        Initialize the membership function.
        
        Args:
            name: Linguistic label for this fuzzy set (e.g., 'Low', 'Medium', 'High')
        """
        self.name = name
    
    @abstractmethod
    def compute(self, x: float) -> float:
        """
        Compute the membership degree for a given input value.
        
        Args:
            x: The crisp input value
            
        Returns:
            Membership degree in the range [0, 1]
        """
        pass
    
    def compute_array(self, x: np.ndarray) -> np.ndarray:
        """
        Compute membership degrees for an array of input values.
        
        Args:
            x: Array of crisp input values
            
        Returns:
            Array of membership degrees
        """
        return np.array([self.compute(val) for val in x])


class TriangularMF(MembershipFunction):
    """
    Triangular membership function.
    
    Defined by three parameters (a, b, c) where:
    - a: Left foot (membership = 0)
    - b: Peak (membership = 1)
    - c: Right foot (membership = 0)
    
    The function rises linearly from a to b, then falls linearly from b to c.
    This is the most commonly used membership function due to its simplicity
    and computational efficiency.
    """
    
    def __init__(self, name: str, a: float, b: float, c: float):
        """
        Initialize triangular membership function.
        
        Args:
            name: Linguistic label for this fuzzy set
            a: Left foot of the triangle (x value where membership becomes > 0)
            b: Peak of the triangle (x value where membership = 1)
            c: Right foot of the triangle (x value where membership becomes 0)
        """
        super().__init__(name)
        if not (a <= b <= c):
            raise ValueError("Parameters must satisfy a <= b <= c")
        self.a = a
        self.b = b
        self.c = c
    
    def compute(self, x: float) -> float:
        """
        Compute triangular membership degree.
        
        Args:
            x: Input value
            
        Returns:
            Membership degree in [0, 1]
        """
        if x <= self.a or x >= self.c:
            return 0.0
        elif x <= self.b:
            # Rising edge
            if self.b == self.a:
                return 1.0
            return (x - self.a) / (self.b - self.a)
        else:
            # Falling edge
            if self.c == self.b:
                return 1.0
            return (self.c - x) / (self.c - self.b)


class TrapezoidalMF(MembershipFunction):
    """
    Trapezoidal membership function.
    
    Defined by four parameters (a, b, c, d) where:
    - a: Left foot (membership = 0)
    - b: Left shoulder (membership = 1)
    - c: Right shoulder (membership = 1)
    - d: Right foot (membership = 0)
    
    The function has a flat top between b and c, making it suitable for
    representing broader categories or ranges.
    """
    
    def __init__(self, name: str, a: float, b: float, c: float, d: float):
        """
        Initialize trapezoidal membership function.
        
        Args:
            name: Linguistic label for this fuzzy set
            a: Left foot
            b: Left shoulder
            c: Right shoulder  
            d: Right foot
        """
        super().__init__(name)
        if not (a <= b <= c <= d):
            raise ValueError("Parameters must satisfy a <= b <= c <= d")
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    
    def compute(self, x: float) -> float:
        """
        Compute trapezoidal membership degree.
        
        Args:
            x: Input value
            
        Returns:
            Membership degree in [0, 1]
        """
        if x <= self.a or x >= self.d:
            return 0.0
        elif x <= self.b:
            # Rising edge
            if self.b == self.a:
                return 1.0
            return (x - self.a) / (self.b - self.a)
        elif x <= self.c:
            # Flat top
            return 1.0
        else:
            # Falling edge
            if self.d == self.c:
                return 1.0
            return (self.d - x) / (self.d - self.c)


class GaussianMF(MembershipFunction):
    """
    Gaussian membership function.
    
    Defined by two parameters (mean, sigma) where:
    - mean: Center of the Gaussian curve (membership = 1)
    - sigma: Standard deviation controlling the width
    
    The Gaussian function provides smooth transitions between states,
    making it suitable for modeling natural phenomena and achieving
    smooth control responses.
    """
    
    def __init__(self, name: str, mean: float, sigma: float):
        """
        Initialize Gaussian membership function.
        
        Args:
            name: Linguistic label for this fuzzy set
            mean: Center of the Gaussian (peak location)
            sigma: Standard deviation (controls width)
        """
        super().__init__(name)
        if sigma <= 0:
            raise ValueError("Sigma must be positive")
        self.mean = mean
        self.sigma = sigma
    
    def compute(self, x: float) -> float:
        """
        Compute Gaussian membership degree.
        
        Args:
            x: Input value
            
        Returns:
            Membership degree in [0, 1]
        """
        return np.exp(-0.5 * ((x - self.mean) / self.sigma) ** 2)


class FuzzySet:
    """
    A fuzzy set defined over a universe of discourse.
    
    Combines a membership function with its domain information to form
    a complete fuzzy set that can be used in fuzzy inference operations.
    """
    
    def __init__(self, name: str, mf: MembershipFunction, 
                 universe_min: float, universe_max: float):
        """
        Initialize a fuzzy set.
        
        Args:
            name: Name of the fuzzy set
            mf: Membership function defining the set
            universe_min: Minimum value of the universe of discourse
            universe_max: Maximum value of the universe of discourse
        """
        self.name = name
        self.mf = mf
        self.universe_min = universe_min
        self.universe_max = universe_max
    
    def membership(self, x: float) -> float:
        """
        Get the membership degree for a value.
        
        Args:
            x: Value to evaluate
            
        Returns:
            Membership degree in [0, 1]
        """
        return self.mf.compute(x)
    
    def get_universe(self, resolution: int = 100) -> np.ndarray:
        """
        Get the universe of discourse as a discrete array.
        
        Args:
            resolution: Number of points in the discretized universe
            
        Returns:
            Array of values spanning the universe of discourse
        """
        return np.linspace(self.universe_min, self.universe_max, resolution)


def create_linguistic_variable(name: str, universe_min: float, universe_max: float,
                               num_terms: int = 5, mf_type: str = 'triangular') -> dict:
    """
    Create a complete linguistic variable with evenly distributed fuzzy sets.
    
    This utility function creates a set of fuzzy sets that cover the entire
    universe of discourse. The standard 5-term set uses:
    NB (Negative Big), NS (Negative Small), ZE (Zero), 
    PS (Positive Small), PB (Positive Big)
    
    Args:
        name: Name of the linguistic variable
        universe_min: Minimum value of universe
        universe_max: Maximum value of universe
        num_terms: Number of linguistic terms (typically 3, 5, or 7)
        mf_type: Type of membership function ('triangular' or 'gaussian')
        
    Returns:
        Dictionary mapping term names to membership functions
    """
    terms = {}
    term_names = {
        3: ['N', 'ZE', 'P'],
        5: ['NB', 'NS', 'ZE', 'PS', 'PB'],
        7: ['NB', 'NM', 'NS', 'ZE', 'PS', 'PM', 'PB']
    }
    
    names = term_names.get(num_terms, 
                           [f'T{i}' for i in range(num_terms)])
    
    step = (universe_max - universe_min) / (num_terms - 1)
    
    for i, term_name in enumerate(names):
        center = universe_min + i * step
        
        if mf_type == 'triangular':
            left = center - step if i > 0 else universe_min
            right = center + step if i < num_terms - 1 else universe_max
            terms[term_name] = TriangularMF(term_name, left, center, right)
        elif mf_type == 'gaussian':
            sigma = step / 2
            terms[term_name] = GaussianMF(term_name, center, sigma)
        else:
            raise ValueError(f"Unknown membership function type: {mf_type}")
    
    return terms
