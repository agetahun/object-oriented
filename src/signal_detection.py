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