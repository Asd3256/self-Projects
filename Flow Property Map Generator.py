import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

# Constants
gamma = 1.4
M_throat = 1.0
M_exit = 2.5
x_throat, y_throat = 0.0, 1.0
num_divisions = 20  # Number of characteristic lines


# Function definitions
def prandtl_meyer(M, gamma=1.4):
    return np.sqrt((gamma + 1) / (gamma - 1)) * np.arctan(
        np.sqrt((gamma - 1) * (M ** 2 - 1) / (gamma + 1))) - np.arctan(np.sqrt(M ** 2 - 1))


def solve_mach(nu_target, gamma=1.4):
    res = root(lambda M: prandtl_meyer(M, gamma) - nu_target, 1.5)
    return float(res.x[0])


def mach_angle(M):
    return np.arcsin(1 / M)


# Initialize plot
plt.figure(figsize=(12, 8))
plt.title(f"Complete Characteristic Network with {num_divisions - 1} C⁺ Segments")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)

# Calculate initial C⁻ lines (throat to centerline)
theta_max = (prandtl_meyer(M_exit) - prandtl_meyer(M_throat)) / 2
theta_values = np.linspace(0, theta_max, num_divisions + 1)[1:]

# Store all points and lines
points = []
c_minus_lines = []
all_c_plus_segments = []

# 1. Create all C⁻ lines from throat to centerline
for i, theta in enumerate(theta_values):
    K_minus = theta + theta  # K⁻ = θ + ν (ν = θ at throat)
    nu_center = K_minus  # At centerline (θ=0), ν = K⁻
    M_center = solve_mach(nu_center)
    mu_center = mach_angle(M_center)

    # Average properties for slope
    theta_avg = theta / 2
    mu_avg = (mach_angle(solve_mach(theta)) + mu_center) / 2
    slope = np.tan(theta_avg - mu_avg)

    # Find centerline intersection
    x_center = x_throat - y_throat / slope

    # Store centerline point
    points.append({
        'type': 'centerline',
        'x': x_center,
        'y': 0,
        'theta': 0,
        'nu': nu_center,
        'M': M_center,
        'mu': mu_center,
        'K_minus': K_minus,
        'line_id': i
    })

    # Store C⁻ line
    c_minus_lines.append({
        'start': (x_throat, y_throat),
        'end': (x_center, 0),
        'theta': theta,
        'K_minus': K_minus,
        'line_id': i
    })

    # Plot C⁻ line
    plt.plot([x_throat, x_center], [y_throat, 0], 'b-', alpha=0.5)

# 2. Calculate all C⁺ segments for each characteristic line
for seg_num in range(num_divisions - 1):
    segment_color = plt.cm.viridis(seg_num / (num_divisions - 1))  # Different color for each segment

    for i in range(len(c_minus_lines) - seg_num - 1):
        # Get starting point (centerline point for first segment, intersection for others)
        if seg_num == 0:
            start_point = points[i]
        else:
            # Find previous intersection point
            start_point = next((p for p in points if p.get('segment') == seg_num - 1 and
                                p.get('source_line') == i), None)
            if not start_point:
                continue

        # Get target C⁻ line
        target_c_minus = c_minus_lines[i + seg_num + 1]

        # K⁺ remains constant for all segments of this characteristic line
        K_plus = start_point['theta'] - start_point['nu']

        # At intersection with target C⁻ line:
        theta_intersect = (target_c_minus['K_minus'] + K_plus) / 2
        nu_intersect = (target_c_minus['K_minus'] - K_plus) / 2
        M_intersect = solve_mach(nu_intersect)
        mu_intersect = mach_angle(M_intersect)

        # Calculate intersection point
        theta_avg = (start_point['theta'] + theta_intersect) / 2
        mu_avg = (start_point['mu'] + mu_intersect) / 2
        slope_c_plus = np.tan(theta_avg + mu_avg)

        # Target C⁻ line equation
        x1, y1 = target_c_minus['start']
        x2, y2 = target_c_minus['end']
        slope_c_minus = (y2 - y1) / (x2 - x1)

        # Find intersection coordinates
        x_intersect = (y1 - slope_c_minus * x1 - start_point['y'] + slope_c_plus * start_point['x']) / (
                    slope_c_plus - slope_c_minus)
        y_intersect = start_point['y'] + slope_c_plus * (x_intersect - start_point['x'])

        # Store intersection point
        intersect_point = {
            'type': 'intersection',
            'x': x_intersect,
            'y': y_intersect,
            'theta': theta_intersect,
            'nu': nu_intersect,
            'M': M_intersect,
            'mu': mu_intersect,
            'K_plus': K_plus,
            'segment': seg_num,
            'source_line': i
        }
        points.append(intersect_point)

        # Store C⁺ segment
        segment_data = {
            'start': (start_point['x'], start_point['y']),
            'end': (x_intersect, y_intersect),
            'M_start': start_point['M'],
            'M_end': M_intersect,
            'segment_num': seg_num,
            'line_num': i,
            'color': segment_color
        }
        all_c_plus_segments.append(segment_data)

        # Plot C⁺ segment
        plt.plot([start_point['x'], x_intersect],
                 [start_point['y'], y_intersect],
                 color=segment_color,
                 linestyle='-',
                 linewidth=1.5,
                 label=f'Segment {seg_num + 1}' if i == 0 else "")

# Plot all points with annotations
for p in points:
    if p['type'] == 'centerline':
        plt.plot(p['x'], p['y'], 'ko', markersize=6)
        plt.text(p['x'], p['y'] + 0.05, f"M={p['M']:.2f}", ha='center', fontsize=8)
    else:
        plt.plot(p['x'], p['y'], 'o', color=all_c_plus_segments[p['segment']]['color'], markersize=6)
        plt.text(p['x'], p['y'] + 0.05, f"M={p['M']:.2f}", ha='center', fontsize=8)

# Plot throat
plt.plot(x_throat, y_throat, 'ro', markersize=8)
plt.text(x_throat, y_throat + 0.1, "Throat (M=1.0)", ha='center')

# Create legend
legend_elements = [
    plt.Line2D([0], [0], color='b', lw=2, label='C⁻ Characteristics'),
    *[plt.Line2D([0], [0], color=all_c_plus_segments[i]['color'], lw=2,
                 label=f'C⁺ Segment {i + 1}') for i in range(num_divisions - 1)],
    plt.Line2D([0], [0], marker='o', color='k', label='Centerline Points',
               markerfacecolor='k', markersize=8, linestyle='None'),
    plt.Line2D([0], [0], marker='o', color='r', label='Throat',
               markerfacecolor='r', markersize=8, linestyle='None')
]

plt.legend(handles=legend_elements, loc='upper right', ncol=2)
plt.tight_layout()

# Print summary data
print(f"\nCharacteristic Network with {num_divisions - 1} C⁺ Segments:")
print(f"{'Type':<12} {'Segment':<8} {'X':<10} {'Y':<10} {'Mach':<6} {'θ (deg)':<8}")
for p in points:
    if p['type'] == 'centerline':
        print(f"{'Centerline':<12} {'N/A':<8} {p['x']:.4f} {p['y']:.4f} {p['M']:.4f} {0:<8.1f}")
    else:
        print(
            f"{'Intersection':<12} {p['segment'] + 1:<8} {p['x']:.4f} {p['y']:.4f} {p['M']:.4f} {np.degrees(p['theta']):<8.1f}")

plt.show()