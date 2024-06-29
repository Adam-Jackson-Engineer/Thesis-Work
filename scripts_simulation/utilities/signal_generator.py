"""
signal_generator.py

Generates various types of signals.
"""

import numpy as np

class SignalGenerator:
    def __init__(self, amplitude=1.0, frequency=0.001, y_offset=0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.y_offset = y_offset

    def square(self, t):    
        if t % (1.0 / self.frequency) <= 0.5 / self.frequency:
            return self.amplitude + self.y_offset
        else:
            return -self.amplitude + self.y_offset
    
    def sawtooth(self, t):
        tmp = t % (0.5 / self.frequency)
        return 4 * self.amplitude * self.frequency * tmp - self.amplitude + self.y_offset
    
    def step(self, t):
        if t >= 0.0:
            return self.amplitude + self.y_offset
        else:
            return self.y_offset

    def random(self, t):
        return np.random.normal(self.y_offset, self.amplitude)

    def sin(self, t):
        return self.amplitude * np.sin(2 * np.pi * self.frequency * t) + self.y_offset
