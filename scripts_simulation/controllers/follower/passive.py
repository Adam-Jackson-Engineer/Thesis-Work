"""
passive.py

A follower controller that will apply no forces, unless there is gravity, then it will
compensate for gravity.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utilities import table_parameters as tp

class Passive:
    """Main controller for the passive follower."""

    def __init__(self, gravity=False):
        """Initialize the Passive controller.

        Args:
            gravity (bool): Flag to indicate if gravity compensation is needed.
        """
        if gravity:
            self.gravity = tp.GRAVITY * tp.TABLE_MASS / 2
        else:
            self.gravity = 0.0

    def update(self, states, u_leader = None):
        """Update the controller to output the appropriate forces, accounting for gravity.

        Args:
            states: The current states of the system (unused in this controller).
        """
        tau = np.array([[0.0],
                        [0.0],
                        [self.gravity],
                        [0.0],
                        [0.0],
                        [0.0]])
        return tau
