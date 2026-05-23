import json
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches

# Load the demo sample data
with open('results/demo_sample.json', 'r') as f:
    data = json.load(f)

tokens = data['n_tokens']
composite = data['composite']
labels = data['labels']

# Create a sequence of token indices
x = np.arange(tokens)

# Set up the plot style for a clean, academic look
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 4), dpi=300)

# Plot the composite score
ax.plot(x, composite, color='#2563eb', linewidth=2.5, label='CAWC v5 Composite Score')

# Fill hallucinated regions
in_hallucination = False
start_idx = 0
for i, label in enumerate(labels):
    if label == 1 and not in_hallucination:
        in_hallucination = True
        start_idx = i
    elif label == 0 and in_hallucination:
        in_hallucination = False
        ax.axvspan(start_idx, i - 0.5, color='#ef4444', alpha=0.2, label='Ground Truth Hallucination' if start_idx == 15 else "")
# Handle case where sequence ends on a hallucination
if in_hallucination:
    ax.axvspan(start_idx, tokens - 1, color='#ef4444', alpha=0.2, label='Ground Truth Hallucination' if 'Ground Truth Hallucination' not in plt.gca().get_legend_handles_labels()[1] else "")


# Add threshold line
ax.axhline(y=0.75, color='#64748b', linestyle='--', linewidth=1.5, label='Detection Threshold (0.75)')

# Formatting
ax.set_xlim(0, tokens - 1)
# Add some padding to y-axis
y_min = min(composite) - 0.1
y_max = max(composite) + 0.1
ax.set_ylim(y_min, y_max)

ax.set_xlabel('Token Position in Generated Sequence', fontsize=12, fontweight='500')
ax.set_ylabel('Composite Hallucination Risk', fontsize=12, fontweight='500')
ax.set_title('Qualitative Example: Token-Level Metric Tracking', fontsize=14, pad=15, fontweight='bold')

# Clean up legend (remove duplicates)
handles, legend_labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(legend_labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper left', frameon=True, framealpha=0.9, edgecolor='#e2e8f0')

plt.tight_layout()
plt.savefig('results/qualitative_example_1.png', bbox_inches='tight')
plt.savefig('results/qualitative_example_1.pdf', bbox_inches='tight') # Also save as PDF for LaTeX
print("Successfully generated results/qualitative_example_1.png and .pdf")
