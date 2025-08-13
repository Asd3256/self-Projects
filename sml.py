import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
import pandas as pd



# Constants
gamma = 1.4
M_throat = 1.0
M_exit = 2.5
x_throat, y_throat = 0.0, 1
num_divisions = 50  # Number of characteristic lines

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
plt.figure(figsize=(14, 8))
plt.title(f"Complete Nozzle Design Using Method of Characteristics\n({num_divisions - 1} C⁺ Lines, Exit Mach {M_exit})",
          fontsize=12)
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Calculate initial C⁻ lines (throat to centerline)
theta_max = (prandtl_meyer(M_exit) - prandtl_meyer(M_throat)) / 2
theta_values = np.linspace(0, theta_max, num_divisions + 1)[1:]

points = []
c_minus_lines = []
all_c_plus_segments = []
last_intersection_points = []

for i, theta in enumerate(theta_values):
    K_minus = theta + theta
    nu_center = K_minus
    M_center = solve_mach(nu_center)
    mu_center = mach_angle(M_center)

    theta_avg = theta / 2
    mu_avg = (mach_angle(solve_mach(theta)) + mu_center) / 2
    slope = np.tan(theta_avg - mu_avg)
    x_center = x_throat - y_throat / slope

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

    c_minus_lines.append({
        'start': (x_throat, y_throat),
        'end': (x_center, 0),
        'theta': theta,
        'K_minus': K_minus,
        'line_id': i
    })

    plt.plot([x_throat, x_center], [y_throat, 0], 'b-', alpha=0.3)

for seg_num in range(num_divisions - 1):
    segment_color = plt.cm.viridis(seg_num / (num_divisions - 1))

    for i in range(len(c_minus_lines) - seg_num - 1):
        if seg_num == 0:
            start_point = points[i]
        else:
            start_point = next((p for p in points if p.get('segment') == seg_num - 1 and
                                p.get('source_line') == i), None)
            if not start_point:
                continue

        target_c_minus = c_minus_lines[i + seg_num + 1]
        K_plus = start_point['theta'] - start_point['nu']

        theta_intersect = (target_c_minus['K_minus'] + K_plus) / 2
        nu_intersect = (target_c_minus['K_minus'] - K_plus) / 2
        M_intersect = solve_mach(nu_intersect)
        mu_intersect = mach_angle(M_intersect)

        theta_avg = (start_point['theta'] + theta_intersect) / 2
        mu_avg = (start_point['mu'] + mu_intersect) / 2
        slope_c_plus = np.tan(theta_avg + mu_avg)

        x1, y1 = target_c_minus['start']
        x2, y2 = target_c_minus['end']
        slope_c_minus = (y2 - y1) / (x2 - x1)

        x_intersect = (y1 - slope_c_minus * x1 - start_point['y'] + slope_c_plus * start_point['x']) / (
                slope_c_plus - slope_c_minus)
        y_intersect = start_point['y'] + slope_c_plus * (x_intersect - start_point['x'])

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

        if seg_num == (num_divisions - 2 - i):
            last_intersection_points.append({
                'x': x_intersect,
                'y': y_intersect,
                'theta': theta_intersect,
                'mu': mu_intersect,
                'line_num': i,
                'segment_num': seg_num
            })

sorted_last_points = sorted(last_intersection_points, key=lambda x: x['line_num'])

wall_points = [{'x': x_throat, 'y': y_throat, 'type': 'throat'}]
theta_max = np.degrees(theta_max)
theta_first = np.degrees(sorted_last_points[0]['theta'])
avg_theta = np.radians((theta_max + theta_first) / 2)
slope = np.tan(avg_theta)

first_segment = next(s for s in all_c_plus_segments if s['line_num'] == 0 and s['segment_num'] == num_divisions - 2)
x1, y1 = first_segment['start']
x2, y2 = first_segment['end']
slope_cplus = (y2 - y1) / (x2 - x1)
intercept_cplus = y1 - slope_cplus * x1

x_wall = (1 - intercept_cplus) / (slope_cplus - slope)
y_wall = slope * x_wall + 1
wall_points.append({'x': x_wall, 'y': y_wall, 'type': 'wall_1', 'theta_avg': avg_theta})

for i in range(1, len(sorted_last_points)):
    prev_point = wall_points[-1]
    prev_theta = np.degrees(sorted_last_points[i - 1]['theta'])
    curr_theta = np.degrees(sorted_last_points[i]['theta'])
    avg_theta = np.radians((prev_theta + curr_theta) / 2)
    slope = np.tan(avg_theta)

    curr_segment = next(s for s in all_c_plus_segments
                        if s['line_num'] == i and s['segment_num'] == num_divisions - 2 - i)
    x1, y1 = curr_segment['start']
    x2, y2 = curr_segment['end']
    slope_cplus = (y2 - y1) / (x2 - x1)
    intercept_cplus = y1 - slope_cplus * x1

    x_wall = (prev_point['y'] - slope * prev_point['x'] - intercept_cplus) / (slope_cplus - slope)
    y_wall = slope * (x_wall - prev_point['x']) + prev_point['y']
    wall_points.append({'x': x_wall, 'y': y_wall, 'type': f'wall_{i + 1}', 'theta_avg': avg_theta})

last_point = c_minus_lines[-1]['end']
wall_points.append({'x': last_point[0], 'y': last_point[1], 'type': 'exit'})
true_wall_points = [p for p in wall_points if p['type'].startswith('wall')]

cplus_to_wall = {}
for i in range(len(true_wall_points)):
    last_segment = [s for s in all_c_plus_segments
                    if s['line_num'] == i and
                    s['segment_num'] == num_divisions - 2 - i][0]
    cplus_to_wall[i] = {
        'cplus_end': last_segment['end'],
        'wall_point': (true_wall_points[i]['x'], true_wall_points[i]['y'])
    }

for segment in all_c_plus_segments:
    plt.plot([segment['start'][0], segment['end'][0]],
             [segment['start'][1], segment['end'][1]],
             color=segment['color'],
             linestyle='-',
             linewidth=1.0,
             alpha=0.7)

for line_num, connection in cplus_to_wall.items():
    x1, y1 = connection['cplus_end']
    x2, y2 = connection['wall_point']
    last_segment_color = [s['color'] for s in all_c_plus_segments
                          if s['line_num'] == line_num and
                          s['segment_num'] == num_divisions - 2 - line_num][0]
    plt.plot([x1, x2], [y1, y2],
             color=last_segment_color,
             linestyle='-',
             linewidth=1.0,
             alpha=0.7)

plt.plot([p['x'] for p in true_wall_points],
         [p['y'] for p in true_wall_points],
         'ro', markersize=5)

plt.plot(x_throat, y_throat, 'ko', markersize=8)
plt.plot(last_point[0], last_point[1], 'ko', markersize=8)
plt.plot(x_throat, -y_throat, 'ko', markersize=8)

plt.plot([p['x'] for p in points if p['type'] == 'centerline'],
         [p['y'] for p in points if p['type'] == 'centerline'],
         'k--', alpha=0.5)

# ====================
# Final clean lines
# ====================
center_pt = [p for p in points if p['type'] == 'centerline'][-1]
x1, y1 = center_pt['x'], center_pt['y']
m1 = np.tan(center_pt['mu'])

last_wall_pt = true_wall_points[-1]
x2, y2 = last_wall_pt['x'], last_wall_pt['y']

last_theta = sorted_last_points[-1]['theta']
avg_theta = last_theta / 2
m2 = np.tan(avg_theta)

x_int = (y2 - y1 + m1 * x1 - m2 * x2) / (m1 - m2)
y_int = m1 * (x_int - x1) + y1

plt.plot([x1, x_int], [y1, y_int], 'g--', linewidth=1.5)
plt.plot([x2, x_int], [y2, y_int], 'r-', linewidth=1.5)
plt.plot(x_int, y_int, 'ro', markersize=4)

# -------------------------------------------------------
# Plot and connect all wall points (including throat)
# -------------------------------------------------------

# Get all wall points including throat
all_wall_x = [p['x'] for p in wall_points if p['type'].startswith('wall') or p['type'] == 'throat']
all_wall_y = [p['y'] for p in wall_points if p['type'].startswith('wall') or p['type'] == 'throat']

# Plot wall points
plt.plot(all_wall_x, all_wall_y, 'ro', markersize=4)

# Connect wall points with a solid red line
plt.plot(all_wall_x, all_wall_y, 'r-', linewidth=1.5)

print("\nAll Wall Points (including throat):")
for i, p in enumerate(wall_points):
    if p['type'].startswith('wall') or p['type'] == 'throat':
        print(f"{i:>2}: X = {p['x']:.6f}, Y = {p['y']:.6f}")


plt.tight_layout()
print(f"\nIntersection Point:\nX = {x_int:.6f}, Y = {y_int:.6f}")

# ===============================
# Lower Contour (Mirror of Upper)
# ===============================
lower_wall_x = all_wall_x
lower_wall_y = [-y for y in all_wall_y]

# Plot lower contour
plt.plot(lower_wall_x, lower_wall_y, 'r-', linewidth=1.5)
plt.plot(lower_wall_x, lower_wall_y, 'ro', markersize=4)

# Print lower wall points
print("\nLower Wall Points (mirrored from upper):")
for i, (x, y) in enumerate(zip(lower_wall_x, lower_wall_y)):
    print(f"{i:>2}: X = {x:.6f}, Y = {y:.6f}")

# ===================================
# Mirror C⁻ and C⁺ Lines for Lower Side
# ===================================

# Mirror C⁻ lines (throat to centerline)
for line in c_minus_lines:
    x1, y1 = line['start']
    x2, y2 = line['end']
    plt.plot([x1, x2], [-y1, -y2], 'b-', alpha=0.3)

# Mirror C⁺ lines (internal characteristics)
for segment in all_c_plus_segments:
    x1, y1 = segment['start']
    x2, y2 = segment['end']
    plt.plot([x1, x2], [-y1, -y2],
             color=segment['color'],
             linestyle='-',
             linewidth=1.0,
             alpha=0.7)

# Mirror last segments from center to wall
for line_num, connection in cplus_to_wall.items():
    x1, y1 = connection['cplus_end']
    x2, y2 = connection['wall_point']
    color = [s['color'] for s in all_c_plus_segments
             if s['line_num'] == line_num and
             s['segment_num'] == num_divisions - 2 - line_num][0]
    plt.plot([x1, x2], [-y1, -y2],
             color=color,
             linestyle='-',
             linewidth=1.0,
             alpha=0.7)
# ===============================
# Final characteristic for lower contour
# ===============================
# Mirror upper points and intersection
# Use correct mirrored centerline point
x_cl, y_cl = center_pt['x'], -center_pt['y']  # mirror y
x_wall_last, y_wall_last = last_wall_pt['x'], -last_wall_pt['y']  # mirror wall point
x_int_lower = x_int
y_int_lower = -y_int


# Plot mirrored lines (same style for symmetry)
plt.plot([x_cl, x_int_lower], [y_cl, y_int_lower], 'g--', linewidth=1.5)
plt.plot([x_wall_last, x_int_lower], [y_wall_last, y_int_lower], 'r-', linewidth=1.5)
plt.plot(x_int_lower, y_int_lower, 'ro', markersize=4)


print(f"\nIntersection Point:\nX = {-x_int:.6f}, Y = {-y_int:.6f}")


# Add converging section using area-Mach relation
def area_mach_relation(M, gamma=1.4):
    return (1 / M) * ((2 / (gamma + 1)) * (1 + ((gamma - 1) / 2) * M**2)) ** ((gamma + 1) / (2 * (gamma - 1)))

# Settings for converging section
M_inlet = 0.2
M_throat = 1.0
x_throat = 0.0
y_throat = 1.0  # Keep consistent with main code

# Generate subsonic Mach numbers from inlet to throat
num_points_converging = 50
mach_subsonic = np.linspace(M_inlet, M_throat, num_points_converging, endpoint=False)

# Calculate area ratios and corresponding y
A_throat = 2 * y_throat
area_ratios = np.array([area_mach_relation(M) for M in mach_subsonic])
areas = area_ratios * A_throat
ys_converging = areas / 2

# Estimate x positions backward from throat assuming a smooth profile
length_converging = 3  # Adjust for visual length
xs_converging = np.linspace(-length_converging, 0, num_points_converging)

# Plot converging wall (upper and lower)
plt.plot(xs_converging, ys_converging, 'r-', linewidth=1.5)
plt.plot(xs_converging, -ys_converging, 'r-', linewidth=1.5)

# Optionally, mark inlet point
plt.plot(xs_converging[0], ys_converging[0], 'ko', markersize=6)
plt.plot(xs_converging[0], -ys_converging[0], 'ko', markersize=6)


plt.show()

# Prepare data for CSV
upper_wall_data = []
lower_wall_data = []

# Add converging section (upper and lower)
for x, y in zip(xs_converging, ys_converging):
    upper_wall_data.append({'Section': 'Upper Converging', 'X': x, 'Y': y})
    lower_wall_data.append({'Section': 'Lower Converging', 'X': x, 'Y': -y})

# Add diverging section (upper and lower)
for x, y in zip(all_wall_x, all_wall_y):
    upper_wall_data.append({'Section': 'Upper Diverging', 'X': x, 'Y': y})
for x, y in zip(lower_wall_x, lower_wall_y):
    lower_wall_data.append({'Section': 'Lower Diverging', 'X': x, 'Y': y})

# Combine data
all_data = upper_wall_data + lower_wall_data

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Save to CSV
csv_filename = "nozzle_wall_new_coordinates_d.csv"
df.to_csv(csv_filename, index=False)
print(f"\nWall point coordinates exported to: {csv_filename}")