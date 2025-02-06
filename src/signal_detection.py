import scipy.stats

class SignalDetection:
    def __init__(self, hits, misses, falseAlarms, correctRejections):
        self.hits = hits
        self.misses = misses
        self.falseAlarms = falseAlarms
        self.correctRejections = correctRejections

    def hit_rate(self):
        """Calculate hit rate with standard correction."""
        total_signal_trials = self.hits + self.misses
        return (self.hits + 0.5) / (total_signal_trials + 1)  # Correction for extreme values

    def false_alarm_rate(self):
        """Calculate false alarm rate with standard correction."""
        total_noise_trials = self.falseAlarms + self.correctRejections
        return (self.falseAlarms + 0.5) / (total_noise_trials + 1)  # Correction for extreme values

    def d_prime(self):
        """Compute d' using Z(hit rate) - Z(false alarm rate)."""
        z_hit = scipy.stats.norm.ppf(self.hit_rate())
        z_false_alarm = scipy.stats.norm.ppf(self.false_alarm_rate())
        return z_hit - z_false_alarm

    def criterion(self):
        """Compute criterion using -0.5 * (Z(hit rate) + Z(false alarm rate))."""
        z_hit = scipy.stats.norm.ppf(self.hit_rate())
        z_false_alarm = scipy.stats.norm.ppf(self.false_alarm_rate())
        return -0.5 * (z_hit + z_false_alarm)

#everything below here is Bayes Factor so change to Signal Detection!
"""
def uniform_prior1(p):
    if p >= 0 and p <= 1:
        return 1.0
    return 0.0

def uniform_prior2(p):
    if p >= 0.45 and p <= 0.55:
        return 10.0
    return 0.0

    # Function to create a uniform prior with arbitrary bounds
def uniform_prior(a, b):
    return lambda p: 1.0 / (b - a) if a <= p <= b else 0.0

    # Example usage with uniform priors over different intervals
prior1 = uniform_prior(0, 1)
prior2 = uniform_prior(0.45, 0.55)

bf = BayesFactor(5, 2, prior1, prior2)
result = bf.compute()
print(result)

bf = BayesFactor(5, 2, uniform_prior1, uniform_prior2)
result = bf.compute()
print(result)
"""
# everything above here is Bayes Factor so change to Signal Detection!

#Signal Detection test required for assignment (taken from Professor's github)
'''
import unittest
import numpy as np
import matplotlib.pyplot as plt
from ..sdt.SignalDetection import SignalDetection

class TestSignalDetection(unittest.TestCase):
    def test_d_prime_zero(self):
        sd   = SignalDetection(15, 5, 15, 5)
        expected = 0
        obtained = sd.d_prime()
        # Compare calculated and expected d-prime
        self.assertAlmostEqual(obtained, expected, places=6)
    def test_d_prime_nonzero(self):
        sd   = SignalDetection(15, 10, 15, 5)
        expected = -0.421142647060282
        obtained = sd.d_prime()
        # Compare calculated and expected d-prime
        self.assertAlmostEqual(obtained, expected, places=6)
    def test_criterion_zero(self):
        sd   = SignalDetection(5, 5, 5, 5)
        # Calculate expected criterion
        expected = 0
        obtained = sd.criterion()
        # Compare calculated and expected criterion
        self.assertAlmostEqual(obtained, expected, places=6)
    def test_criterion_nonzero(self):
        sd   = SignalDetection(15, 10, 15, 5)
        # Calculate expected criterion
        expected = -0.463918426665941
        obtained = sd.criterion()
        # Compare calculated and expected criterion
        self.assertAlmostEqual(obtained, expected, places=6)

if __name__ == '__main__':
    unittest.main()
    '''