from scipy.stats import norm
import numpy as np

class SignalDetection:
    def __init__(self, hits, misses, falseAlarms, correctRejections):
        self.hits = hits
        self.misses = misses
        self.falseAlarms = falseAlarms
        self.correctRejections = correctRejections
    
    def hit_rate(self):
        return self._adjust_rate(self.hits, self.hits + self.misses)
    
    def false_alarm_rate(self):
        return self._adjust_rate(self.falseAlarms, self.falseAlarms + self.correctRejections)
    
    def d_prime(self):
        return norm.ppf(self.hit_rate()) - norm.ppf(self.false_alarm_rate())
    
    def criterion(self):
        return -0.5 * (norm.ppf(self.hit_rate()) + norm.ppf(self.false_alarm_rate()))
    
    def _adjust_rate(self, count, total):
        """Adjusts rates to avoid extreme values of 0 or 1."""
        adjusted = (count + 0.5) / (total + 1)
        return np.clip(adjusted, 1e-6, 1 - 1e-6)

#everything below here is Bayes Factor so change to Signal Detection!
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

import unittest

class TestBayesFactor(unittest.TestCase):  
    
    def test_init(self):
        n = 5
        k = 2
        bf = BayesFactor(n, k, uniform_prior1, uniform_prior2)
        self.assertEqual(bf.n, n)
        self.assertEqual(bf.k, k)
        self.assertEqual(bf.prior1, uniform_prior1)
        self.assertEqual(bf.prior2, uniform_prior2)

    def test_compute(self):
        n = 5
        k = 2
        bf = BayesFactor(n, k, uniform_prior1, uniform_prior1)

        self.assertEqual(bf.compute(), 1.)
          
# Run the tests
if __name__ == '__main__':
    unittest.main()