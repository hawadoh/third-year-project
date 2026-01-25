from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import ConnectionPatch
import numpy as np

from spm.data.samplers import euclidean_depths

# Use logs_mps directory instead of ANALYSIS_DIR
ANALYSIS_DIR = Path(__file__).parent.parent / "logs_mps"


def load_data():
    exp_to_verifiability = defaultdict(list)
    exp_to_correctness = defaultdict(list)
    # for each .csv in ANALYSIS_DIR, load the csv and grab the last row penultimate column (transcript accuracy)
    for path in ANALYSIS_DIR.glob('*.csv'):
        experiment_name = path.stem.split('-')[1]
        exp_name, seed = experiment_name.split('_')[0], experiment_name[-1]
        with open(path) as f:
            if "TL" not in experiment_name:  # It's the Baseline
                continue
            for line in f:  # go to the last line
                pass
            final_scores = line.strip().split(',')
            verifiability = float(final_scores[-2])
            exp_to_verifiability[exp_name].append(verifiability)
            exp_to_correctness[exp_name].append(float(line.strip().split(',')[1]))

    score_to_draw = exp_to_verifiability  # change to correctness if desired (requires a few tweaks further)
    scores = {"T=0" if k == "TL" else f"T={k[-1]}": v for k, v in score_to_draw.items()}
    keys_sorted = sorted(scores.keys())  # This puts T=0 first, then T=1, T=2, etc.
    means = [np.mean(scores[k]) for k in keys_sorted]
    stderrs = [np.std(scores[k]) / np.sqrt(len(scores[k])) for k in keys_sorted]
    return keys_sorted, means, stderrs


def draw_errorbar(ax, mean, stderr, color, text):
    ax.errorbar(mean, y=0, xerr=stderr, fmt='o', color=color, capsize=5)
    if text is not None:
        # Place T=4 label below the point
        y_pos = -0.0175 if text == "T=4" else 0.01
        ax.text(mean, y_pos, text, ha='center', color=color)


def draw_depth(ax, depth, color, text, zoomed):
    # Add a dashed vertical line to show the depth
    # make line length shorter
    ymax = 0.9 if zoomed else 0.35
    ax.axvline(x=depth, ymax=ymax, color=color, linestyle='--')
    if text is None:
        return
    x = depth + 0.005 if zoomed else depth + 0.012
    y = -0.05 if zoomed else 0.02
    ax.text(x, y, text, ha='center', color=color, rotation=90, va='bottom')


def draw(ax, zoomed_ax, keys_sorted, means, stderrs, depths, zoomed=False):
    # Full color scheme: T=0, T=2, T=3, T=4, T=5, T=6, T=7
    all_colors = ['#0077BB', '#009988', '#EE7733', '#CC3311', "#EE3377", "#33BBEE", "#DDAA33"]
    # Skip first color (used for depth 0) since mps data starts at T=2
    colors = all_colors[1:]
    ax_pos = ax.get_position()
    ydelta = 0.23
    lower_ax = fig.add_axes([ax_pos.x0, ax_pos.y0 - ydelta, ax_pos.width, ax_pos.height - ydelta], sharex=ax,
                            frameon=False)
    for exp, mean, stderr, color, depth in zip(keys_sorted, means, stderrs, colors, depths):
        print(f"{exp}: {mean} ± {stderr:.2f} (depth: {depth})")
        if mean <= ZOOM_LOW:
            draw_errorbar(lower_ax, mean, stderr, color, exp)
        else:
            lower_ax.plot(mean, 0, 'o', color=color)
            draw_errorbar(zoomed_ax, mean, stderr, color, exp)
        # Always label depths on the main (lower) graph
        draw_depth(ax, depth, color, f"Euclidean depth {exp[-1]}", zoomed=False)
        if depth > ZOOM_LOW:
            # Also show in zoomed view
            draw_depth(zoomed_ax, depth, color, f"Euclidean depth {exp[-1]}", zoomed=True)
    # Draw final depth (one more than the last T)
    draw_depth(ax, depths[-1], 'black', f'Euclidean depth {Ts[-1]}', zoomed=False)
    draw_depth(zoomed_ax, depths[-1], 'black', f'Euclidean depth {Ts[-1]}', zoomed=True)
    # Setting labels and title
    ax.set_xlabel('Verifiability')
    ax.set_yticks([])  # Remove y-axis ticks
    zoomed_ax.set_yticks([])  # Remove y-axis ticks
    lower_ax.set_yticks([])  # Remove y-axis ticks
    # Format the x-axis to show percentages
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    zoomed_ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))


keys_sorted, means, stderrs = load_data()
Ts = [int(k[-1]) for k in keys_sorted]
Ts += [Ts[-1] + 1]  # Add one more depth for the zoomed in plot
depths = euclidean_depths(Ts)
# Compute depth 0 separately (not tied to any experiment)
depth_0 = euclidean_depths([0])[0]

# Zoom range: 61% to 77%
ZOOM_LOW = 0.61
ZOOM_HIGH = 0.77

# Draw main plot
fig, ax = plt.subplots(figsize=(8, 5))

# Create inset of zoomed in region
zoomed_bbox = [0.2, 0.5, 0.7, 0.3]  # Adjust position and size of inset
inset_ax = fig.add_axes(zoomed_bbox)
inset_ax.set_xlim(ZOOM_LOW, ZOOM_HIGH)
ax.set_xlim(left=0.15, right=1.01)  # Extend to 100% so it appears on axis
ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))  # Ticks every 10%
draw(ax, inset_ax, keys_sorted, means, stderrs, depths, zoomed=True)

# Draw Euclidean depth 0 on main graph (use first color from scheme to match T=0)
draw_depth(ax, depth_0, '#0077BB', 'Euclidean depth 0', zoomed=False)

# draw dashed lines connecting main axis to inset corners using ConnectionPatch
# Left line: from ZOOM_LOW on main axis to bottom-left corner of inset
con_left = ConnectionPatch(xyA=(ZOOM_LOW, 0), xyB=(0, 0), coordsA="data", coordsB="axes fraction",
                           axesA=ax, axesB=inset_ax, color='black', linestyle='dotted')
fig.add_artist(con_left)
# Right line: from ZOOM_HIGH on main axis to bottom-right corner of inset
con_right = ConnectionPatch(xyA=(ZOOM_HIGH, 0), xyB=(1, 0), coordsA="data", coordsB="axes fraction",
                            axesA=ax, axesB=inset_ax, color='black', linestyle='dotted')
fig.add_artist(con_right)
# Set zoomed x-axis ticks every 3%
inset_ax.xaxis.set_major_locator(mticker.MultipleLocator(0.03))
inset_ax.text(0.98, 1.15, "(Zoomed-in)", ha='right', va='top', transform=inset_ax.transAxes)

# Adjusting the layout to save space
# plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)

# No box around the plot, only x axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

format = "pdf"
plt.savefig(f'annotation-mps.{format}', dpi=1200, format=format)
plt.show()
