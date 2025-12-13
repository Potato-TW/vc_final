import numpy as np
from typing import Optional

class Zigzag:
    """
    Utility to generate and retrieve Zigzag scan patterns.
    """
    _pattern_cache: dict = {}

    @classmethod
    def get_zigzag(cls, n: int = 8) -> np.ndarray:
        """
        Get the zigzag pattern for an nxn block. 
        Uses caching to avoid recomputing for every image.
        """
        if n not in cls._pattern_cache:
            cls._pattern_cache[n] = cls._create_zigzag_pattern(n)
        return cls._pattern_cache[n]

    @staticmethod
    def _create_zigzag_pattern(n: int) -> np.ndarray:
        """
        Efficiently creates zigzag pattern indices for nxn matrix.
        """
        # Initialize an nxn matrix with the zigzag sequence
        pattern = np.zeros((n, n), dtype=int)
        
        i, j = 0, 0
        for k in range(n * n):
            pattern[i, j] = k
            
            # Decide next direction based on sum of coords (parity)
            if (i + j) % 2 == 0:  # Moving Up-Right
                if j == n - 1:    # Hit right wall
                    i += 1
                elif i == 0:      # Hit top wall
                    j += 1
                else:
                    i -= 1
                    j += 1
            else:                 # Moving Down-Left
                if i == n - 1:    # Hit bottom wall
                    j += 1
                elif j == 0:      # Hit left wall
                    i += 1
                else:
                    i += 1
                    j -= 1

        # Flatten and return coordinates sorted by the sequence value
        flat_indices = np.argsort(pattern.flatten())
        rows, cols = np.unravel_index(flat_indices, (n, n))
        return np.column_stack((rows, cols))