import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

# read Harris Part I Data
harris_part1 =pd.read_csv('HarrisPartI.csv')

# read Harris Part II Data
harris_part2 =pd.read_csv('HarrisPartIII.csv')

# read Krause 21 Data
Krause21 =pd.read_csv('Krause21.csv')

# read vandenBerg_Table2
vandenBerg_table2 =pd.read_csv('vandenBerg_table2.csv')

# Defining Variables for Krause Data
Age_Krause = Krause21['Age']
Name_Krause = Krause21['Object']
FeH_Krause = Krause21['FeH']
Names_Krause = Krause21['Object']
# Make "NGC" followed by a space
Names_Krause = [name.replace("NGC", "NGC ") if name.startswith("NGC") and not name.startswith("NGC ") else name for name in Names_Krause]
Stellar_Mass_Krause = Krause21['Mstar']

# Defining Variables for Van Den Berg Data
FeH_vdb = vandenBerg_table2['FeH']
Age_vdb = vandenBerg_table2['Age']
Names_vdb = vandenBerg_table2['#NGC']
# Add NGC to the name
Names_vdb = [f"NGC {names}" for names in Names_vdb]
Galcen_vdb = vandenBerg_table2['R_G']
Age_error_vdb = vandenBerg_table2['Age_err']

# Defining Variables for Harris Data
Cluster_r_Harris = harris_part2['r_c']
v_disp = harris_part2['sig_v']
Names_Harris = harris_part2['ID']
X_Harris = harris_part1['X']
Y_Harris = harris_part1['Y']
Z_Harris = harris_part1['Z']
Rotational_v_Harris = harris_part2['v_r']
R_gc_Harris = harris_part1['R_gc']
l_Harris = np.radians(harris_part1['L'])
b_Harris = np.radians(harris_part1['B'])
v_LSR_Harris = harris_part2['v_LSR']

### =================== KRAUSE AGE vs METALICITY =================== ###

# scatter plot Krause Age vs Metalicity
plt.scatter(FeH_Krause, Age_Krause, c = Age_Krause, cmap = 'coolwarm')

#Add labels and titles for Krause Metalicity plot
for i in range(len(Krause21)):
    plt.text(FeH_Krause[i] + 0.05 * np.max(FeH_Krause) / len(Krause21),   #small x-offset
             Age_Krause[i] + 0.05 * np.max(Age_Krause) / len(Krause21),   # small y-offset
             Names_Krause[i], fontsize=7, color='black', alpha=0.8)

#Add labels and titles for the plot
plt.xlabel('Metalicity [Fe/H]')
plt.ylabel('Age (Gyrs)')
plt.title('Age vs Metalicity (Krause)')

plt.scatter(FeH_Krause, Age_Krause, c=Age_Krause, cmap='coolwarm')

# Labels (keep as in your code)
for i in range(len(Krause21)):
    plt.text(
        FeH_Krause[i] + 0.05 * np.max(FeH_Krause) / len(Krause21),
        Age_Krause[i] + 0.05 * np.max(Age_Krause) / len(Krause21),
        Names_Krause[i], fontsize=7, color='black', alpha=0.8
    )

# -------- helpers --------
def find_idx(names, key):
    key_norm = key.lower().replace(" ", "")
    for i, n in enumerate(names):
        if n.lower().replace(" ", "") == key_norm:
            return i
    for i, n in enumerate(names):
        if key_norm in n.lower().replace(" ", ""):
            return i
    return None

def weighted_flat_quad_fit(xi, yi, x0, w=None, min_abs_a=None):
    """
    Fit y = c + a*(x-x0)**2 with a<=0 (concave-down) and slope 0 at x0.
    Weighted least squares; optionally enforce |a| >= min_abs_a.
    """
    if w is None:
        w = np.ones_like(xi)
    A = np.vstack([np.ones_like(xi), (xi - x0)**2]).T
    # weighted solve without building a big diagonal
    Aw = A * w[:, None]
    yw = yi * w
    c, a = np.linalg.lstsq(Aw, yw, rcond=None)[0]
    a = -abs(a)  # force concave-down
    if (min_abs_a is not None) and (abs(a) < min_abs_a):
        a = -min_abs_a
    return lambda xq: c + a*(xq - x0)**2, a, c

# -------- indices of steering targets --------
ix_6388  = find_idx(Names_Krause, "NGC 6388")
ix_4590  = find_idx(Names_Krause, "NGC 4590")
ix_palom = find_idx(Names_Krause, "Palomar 12")

x = FeH_Krause
y = Age_Krause
x0 = np.min(x)  # flat-at-left anchor (lowest metallicity)

# -------- split sequences --------
top_mask    = y >= np.quantile(y, 0.60)
bottom_mask = y <= np.quantile(y, 0.35)

# -------- weights to steer curves --------
w_top = np.ones_like(y, dtype=float)
if ix_6388 is not None: w_top[ix_6388] = 30.0

w_bot = np.ones_like(y, dtype=float)
if ix_4590 is not None: w_bot[ix_4590] = 25.0
if ix_palom is not None: w_bot[ix_palom] = 35.0

# -------- fits --------
f_top, a_top, c_top = weighted_flat_quad_fit(x[top_mask], y[top_mask], x0, w=w_top[top_mask])
f_bot, a_bot, c_bot = weighted_flat_quad_fit(
    x[bottom_mask], y[bottom_mask], x0, w=w_bot[bottom_mask],
    min_abs_a=1.20*abs(a_top)  # make bottom fall faster
)

# -------- shift bottom curve upward to NGC 4590 --------
if ix_4590 is not None:
    shift = y[ix_4590] - f_bot(x[ix_4590])  # match NGC 4590 exactly
else:
    shift = 0.3  # fallback tweak if not found
f_bot_shifted = lambda xq: f_bot(xq) + shift

# -------- plot curves --------
xx = np.linspace(x.min(), x.max(), 400)
plt.plot(xx, f_top(xx),         color='red',        lw=2.3, label='In Situ')
plt.plot(xx, f_bot_shifted(xx), color='lightcoral', lw=2.3, label='Accreted')

# Optional: highlight the steering targets
for idx in [ix_6388, ix_4590, ix_palom]:
    if idx is not None:
        plt.scatter(x[idx], y[idx], s=70, facecolors='none', edgecolors='k')

plt.xlabel('Metallicity [Fe/H]')
plt.ylabel('Age (Gyrs)')
plt.title('Age vs Metalicity (Krause)')
plt.legend()
# Identify likely accreted clusters
# ----------------------------
# A cluster is "accreted" if it lies closer to the bottom curve than to the top
y_top_pred = f_top(FeH_Krause)
y_bot_pred = f_bot_shifted(FeH_Krause)

# Distance from each cluster to each curve
dist_top = np.abs(Age_Krause - y_top_pred)
dist_bot = np.abs(Age_Krause - y_bot_pred)

# Tag clusters closer to bottom curve
accreted_mask = dist_bot < dist_top
n_accreted = np.sum(accreted_mask)

print(f"Estimated number of accreted clusters: {n_accreted}")

# (Optional) visually highlight them
plt.scatter(
    FeH_Krause[accreted_mask],
    Age_Krause[accreted_mask],
    facecolors='none', edgecolors='blue', s=70, linewidths=1.5,
    label='Likely accreted clusters'
)

#Show plot
plt.show()

### =================== VAN DEN BERG AGE vs METALICITY =================== ###

plt.figure(figsize=(8,6))
plt.errorbar(FeH_vdb, Age_vdb,
             yerr=Age_error_vdb, fmt='none',  # error bars
             ecolor='grey', elinewidth=1, capsize=2, alpha=0.6, zorder=1)

# scatter plot Age vs Metalicity Van Den Berg
plt.scatter(FeH_vdb, Age_vdb, c=Age_vdb, cmap='coolwarm', s=40, zorder=2)
for i in range(len(vandenBerg_table2)):
    plt.text(FeH_vdb[i] + 0.05 * np.max(FeH_vdb) / len(vandenBerg_table2),   #small x-offset
             Age_vdb[i] + 0.05 * np.max(Age_vdb) / len(vandenBerg_table2),   # small y-offset
             Names_vdb[i], fontsize=7, color='black', alpha=0.8)

#Add labels (names) next to each data point
for i in range(len(vandenBerg_table2)):
    plt.text(Age_vdb[i] + 0.02*np.max(Age_vdb),   # small x-offset
             FeH_vdb[i] + 0.02*np.max(FeH_vdb),  # small y-offset
             Names_vdb[i], fontsize=7, color='black', alpha=0.8)

#Add labels and titles for the plot
plt.xlabel('Metalicity [Fe/H]')
plt.ylabel('Age (Gyrs)')
plt.title('Age vs Metalicity (Van Den Berg)')

# ---------- helper ----------
def polyfit_weighted(x, y, deg=3, w=None):
    if w is None:
        return np.polyfit(x, y, deg)
    return np.polyfit(x, y, deg, w=w)

def smooth_xy_for_plot(x):
    return np.linspace(x.min()-0.05, x.max()+0.05, 500)

x = FeH_vdb
y = Age_vdb

# Global quadratic residual split
p_all = np.polyfit(x, y, deg=2)
resid = y - np.polyval(p_all, x)
mask_upper = resid >= 0
mask_lower = ~mask_upper

# UPPER BRANCH (In Situ)
p_up = np.polyfit(x[mask_upper], y[mask_upper], 2)
xx_up = smooth_xy_for_plot(x[mask_upper])
yy_up = np.polyval(p_up, xx_up)
plt.plot(xx_up, yy_up, color="#d32f2f", lw=2.8, label="In Situ", zorder=2.6)

# LOWER BRANCH (Accreted) 
x_lo, y_lo = x[mask_lower], y[mask_lower]
w_lo = np.ones_like(y_lo)

# Strongly attract curve toward [Fe/H]≈−0.7 (around NGC XXXX)
target_feh = -0.7
anchor_x = np.array([target_feh])
anchor_y = np.array([8.8])  
anchor_w = np.array([15.0]) 

# Boost upper part to keep flatter top
q75 = np.quantile(y_lo, 0.75)
w_lo[y_lo >= q75] *= 2.0

# Combine with anchor
x_fit = np.concatenate([x_lo, anchor_x])
y_fit = np.concatenate([y_lo, anchor_y])
w_fit = np.concatenate([w_lo, anchor_w])

# Fit cubic and plot
p_lo = polyfit_weighted(x_fit, y_fit, deg=2, w=w_fit)
xx_lo = smooth_xy_for_plot(x_lo)
yy_lo = np.polyval(p_lo, xx_lo)
plt.plot(xx_lo, yy_lo, color="#f6a5a5", lw=2.8, ls="--", label="Potentially Accreted", zorder=2.6)

plt.legend(loc="lower left", frameon=False)

# ----- VAN DEN BERG VERSION -----
try:
    names = np.array(Names_vdb)      # cluster names
    x, y  = FeH_vdb, Age_vdb         # metallicity, age

    accreted_mask = mask_lower       # your dotted branch
    insitu_mask   = mask_upper       # your solid branch

    accreted_names_vdb = names[accreted_mask]
    insitu_names_vdb   = names[insitu_mask]

    print("\nAccreted candidates (van den Berg ages):")
    for n in accreted_names_vdb: print(" -", n)

    # Optional: save
    # import pandas as pd
    # pd.DataFrame({"cluster": accreted_names_vdb}).to_csv("accreted_candidates_vdb.csv", index=False)
except Exception as e:
    print("VdB list not produced:", e)

#Show plot
plt.show()


### =================== HARRIS 3D POSITION =================== ###

# Sun position
R0 = 8.2 # kpc, Galactocentric radius of the Sun
z0 = 0.02 # kpc, Sun height above the midplane (~20 pc)

# Convert to Galactocentric
X_gc = R0 - X_Harris
Y_gc = Y_Harris
Z_gc = z0 + Z_Harris

# 3D position 
plt.figure(1)
ax = plt.axes(projection='3d')
# Some data points black so they are visible on the plot
sc = ax.scatter(X_gc, Y_gc, Z_gc)

# Create a circle (radius 10 kpc, centered at origin)
r = 10  # radius in kpc
theta = np.linspace(0, 2*np.pi, 200)
x_circle = r * np.cos(theta)
y_circle = r * np.sin(theta)
z_circle = np.zeros_like(theta)
ax.plot(x_circle, y_circle, z_circle, color='black', alpha=0.4, linewidth=2, label='10 kpc circle')
ax.plot_trisurf(x_circle, y_circle, z_circle, color='red', alpha=0.1, linewidth=0)

# Create a sphere (radius 3 kpc, centered at origin)
r_sphere = 3  # radius in kpc
phi = np.linspace(0, np.pi, 100) # polar angle
theta_sphere = np.linspace(0, 2*np.pi, 100) # azimuthal angle
theta_sphere, phi = np.meshgrid(theta_sphere, phi)
x_sphere = r_sphere * np.sin(phi) * np.cos(theta_sphere)
y_sphere = r_sphere * np.sin(phi) * np.sin(theta_sphere)
z_sphere = r_sphere * np.cos(phi)
ax.plot_surface(x_sphere, y_sphere, z_sphere, color='green', alpha=0.15, linewidth=0, zorder=1)
ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='black', alpha=0.3, linewidth=0.4, zorder=2)

ax.set_xlabel('x Displacement (kpc)')
ax.set_ylabel('y Displacement (kpc)')
ax.set_zlabel('z Displacement (kpc)')

# Show plot
plt.show()

# Zoomed in plot closer to center to see circle more clearly

lim = 15.0 # axis limits for x, y, z

# Compute mask for points inside the 10 kpc circle in the x–y plane
dist_xy = np.sqrt(X_gc**2 + Y_gc**2)
inside_mask = dist_xy <= r

# Plotting zoomed in figure of previous graph
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_gc, Y_gc, Z_gc)

# Draw the 10 kpc circle in the x–y plane
theta = np.linspace(0, 2*np.pi, 300)
ax.plot(x_circle, y_circle, z_circle, color='red', alpha=0.7, linewidth=1.8)
ax.plot_trisurf(x_circle, y_circle, z_circle, color='red', alpha=0.1, linewidth=0)

# 3 kpc sphere (blue)
ax.plot_surface(x_sphere, y_sphere, z_sphere, color='green', alpha=0.15, linewidth=0, zorder=1)
ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='black', alpha=0.3, linewidth=0.4, zorder=2)

# Axes labels and titles
ax.set_xlabel('x Displacement (kpc)')
ax.set_ylabel('y Displacement (kpc)')
ax.set_zlabel('z Displacement (kpc)')
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)

plt.show()

# 2d plot of x-y plan to more easily identify accreted clusters
ax.scatter(X_gc, Y_gc, s=30)
outside_mask = ~inside_mask
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(X_gc, Y_gc, s=30)

# Circle
circle = plt.Circle((0, 0), r, edgecolor='red', linewidth=1.6, facecolor='red', alpha=0.3, label='10 kpc circle')
ax.add_patch(circle)

# Label clusters outside of the circle
for i in range(len(Names_Harris)):
    if outside_mask[i]:
        ax.text(X_gc[i], Y_gc[i], str(Names_Harris[i]), fontsize=8, color='black', alpha=0.9)

# Axes labels, equal aspect, limits
ax.set_xlabel('x Displacement (kpc)')
ax.set_ylabel('y Displacement (kpc)')
ax.set_title('x–y Projection with 10 kpc Circle')
ax.set_aspect('equal', adjustable='box')

# Set limits to include data and the circle
max_extent = 1.05 * np.max(np.abs(np.r_[X_Harris, Y_Harris, r]))
ax.set_xlim(-max_extent, max_extent)
ax.set_ylim(-max_extent, max_extent)

ax.legend()
plt.show()

### =================== HARRIS CORE RADIUS vs VELOCITY DISPERSION =================== ###

# scatter plot Harris Part III Data
plt.scatter(Cluster_r_Harris, v_disp, c = Cluster_r_Harris, cmap = 'coolwarm')

#Add labels (names) next to each data point
for i in range(len(harris_part2)):
    plt.text(Cluster_r_Harris[i] + 0.02*np.max(Cluster_r_Harris),   # small x-offset
             v_disp[i] + 0.02*np.max(v_disp),  # small y-offset
             Names_Harris[i], fontsize=7, color='black', alpha=0.8)

#Add labels and titles for the plot
plt.xlabel('Core Radius (arcmin)')
plt.ylabel('Velocity Dispersion (km/s)')
plt.title('Core Radius vs Velocity Dispersion (Harris)')

#Show plot
plt.show()

### =================== HARRIS GALACTOCENTRIC RADIUS vs HEIGHT ABOVE GALACTIC PLANE ===================

# Galactocentric Radius vs Hight above Galactic Plane
plt.scatter(R_gc_Harris, abs(Z_Harris), c=R_gc_Harris, cmap='coolwarm')

for i in range(len(vandenBerg_table2)):
    plt.text(R_gc_Harris[i] + 0.05 * np.max(R_gc_Harris) / len(vandenBerg_table2),   #small x-offset
             abs(Z_Harris[i]) + 0.05 * np.max(abs(Z_Harris)) / len(vandenBerg_table2),   # small y-offset
             Names_vdb[i], fontsize=7, color='black', alpha=0.8)

# Draw a line at x-axis 10 kpc
plt.axvline(x=10, color='red', linestyle='--', linewidth=1.2, label='10 kpc')

# Axis labels and titles
plt.xlabel('Galactocentric Radius (kpc)')
plt.ylabel('Height Above Galactic Plane (kpc)')
plt.title('Galactocentric Radius vs Height Above Galactic Plane (Harris)')

plt.show()

### =================== KRAUSE AGE vs STELLAR MASS =================== ###

#Scatter plot Mass v.s. Age
plt.scatter(Age_Krause, Stellar_Mass_Krause, c = Age_Krause, cmap = 'coolwarm')

#Add labels (names) next to each data point
for i in range(len(Krause21)):
    plt.text(Age_Krause[i] + 0.05 * np.max(Age_Krause) / len(Krause21),   #small y-offset
             Stellar_Mass_Krause[i] + 0.05 * np.max(Stellar_Mass_Krause) / len(Krause21), # small x-offset
             Names_Krause[i], fontsize=7, color='black', alpha=0.8)

#Add lables and titles for plot
plt.xlabel('Stellar Age (Gyrs)')
plt.ylabel('Stellar Mass ($M_\odot$)')
plt.title('Stellar Age vs Stellar Mass (Krause)')

#Show plot
plt.show()

### =================== VAN DEN BERG METALICITY vs GALACTOCENTRIC RADIUS =================== ###

# scatter plot Van Der Berg Data
plt.scatter(Galcen_vdb, FeH_vdb, c = Galcen_vdb, cmap = 'coolwarm')

#Add labels (names) next to each data point
for i in range(len(vandenBerg_table2)):
    plt.text(Galcen_vdb[i] + 0.02*np.max(Galcen_vdb),   # small x-offset
             FeH_vdb[i] + 0.02*np.max(FeH_vdb),  # small y-offset
             Names_vdb[i], fontsize=7, color='black', alpha=0.8)

#Add labels and titles for the plot
plt.xlabel('Galactocentric Radius (kpc)')
plt.ylabel('Metalicity [Fe/H]')
plt.title('Metalicity vs Galactocentric Radius (Van Den Berg)')

#Show plot
plt.show()