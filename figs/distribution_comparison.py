"""
Generate comparison figure: uniform vs log-uniform distribution.
Shows why models trained on log-uniform fail on uniformly sampled inputs.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Match the actual sampling parameters from spm/data/samplers.py
UBOUND = 10**4
BASE = 10
NUM_SAMPLES = 100_000

def sample_loguniform(n):
    """Log-uniform sampling as done in LogUniformGCDSampler."""
    log_ubound = np.log10(UBOUND)
    log_vals = np.random.uniform(0, log_ubound, size=n)
    return np.round(np.power(BASE, log_vals)).astype(int)

def sample_uniform(n):
    """Uniform sampling over [1, UBOUND]."""
    return np.random.randint(1, UBOUND + 1, size=n)

np.random.seed(67)
loguniform_samples = sample_loguniform(NUM_SAMPLES)
uniform_samples = sample_uniform(NUM_SAMPLES)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Log-spaced bins for log x-axis
bins = np.logspace(0, 4, 50)

# Log-uniform distribution (flat on log scale)
ax1.hist(loguniform_samples, bins=bins, color='#0077BB', alpha=0.8, edgecolor='white',
         weights=np.ones(len(loguniform_samples)) / len(loguniform_samples) * 100)
ax1.set_xscale('log')
ax1.set_xlabel('Value')
ax1.set_ylabel('Percentage of samples (%)')
ax1.set_title('Log-uniform distribution')
ax1.set_xlim(1, UBOUND)

# Uniform distribution (right-skewed on log scale)
ax2.hist(uniform_samples, bins=bins, color='#EE7733', alpha=0.8, edgecolor='white',
         weights=np.ones(len(uniform_samples)) / len(uniform_samples) * 100)
ax2.set_xscale('log')
ax2.set_xlabel('Value')
ax2.set_ylabel('Percentage of samples (%)')
ax2.set_title('Uniform distribution')
ax2.set_xlim(1, UBOUND)

# Match y-axis scales for fair comparison
max_y = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
ax1.set_ylim(0, max_y)
ax2.set_ylim(0, max_y)

plt.tight_layout()
plt.savefig('distribution_comparison.pdf', format='pdf')
print("Saved distribution_comparison.pdf")
