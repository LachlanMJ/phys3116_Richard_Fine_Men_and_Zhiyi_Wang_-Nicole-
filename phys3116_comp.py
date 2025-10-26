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

### =================== KRAUSE AGE vs METALICITY =================== ###

# scatter plot Krause Age vs Metalicity
plt.scatter(Age_Krause, FeH_Krause, c = Age_Krause, cmap = 'coolwarm')

#Add labels and titles for Krause Metalicity plot
for i in range(len(Krause21)):
    plt.text(Age_Krause[i] + 0.05 * np.max(Age_Krause) / len(Krause21),   #small x-offset
             FeH_Krause[i] + 0.05 * np.max(FeH_Krause) / len(Krause21),   # small y-offset
             Names_Krause[i], fontsize=7, color='black', alpha=0.8)

#Add labels and titles for the plot
plt.xlabel('Age (Gyrs)')
plt.ylabel('Metalicity [Fe/H]')
plt.title('Age vs Metalicity (Krause)')

#Show plot
plt.show()

### =================== VAN DEN BERG AGE vs METALICITY =================== ###

plt.figure(figsize=(8,6))
plt.errorbar(Age_vdb, FeH_vdb,
             xerr=Age_error_vdb, fmt='none',  # no markers, just error bars
             ecolor='grey', elinewidth=1, capsize=2, alpha=0.6, zorder=1)

# scatter plot Age vs Metalicity Van Den Berg
plt.scatter(Age_vdb, FeH_vdb, c=Age_vdb, cmap='coolwarm', s=40, zorder=2)
for i in range(len(vandenBerg_table2)):
    plt.text(Age_vdb[i] + 0.05 * np.max(Age_vdb) / len(vandenBerg_table2),   #small x-offset
             FeH_vdb[i] + 0.05 * np.max(FeH_vdb) / len(vandenBerg_table2),   # small y-offset
             Names_vdb[i], fontsize=7, color='black', alpha=0.8)

#Add labels (names) next to each data point
for i in range(len(vandenBerg_table2)):
    plt.text(FeH_vdb[i] + 0.02*np.max(FeH_vdb),   # small x-offset
             Age_vdb[i] + 0.02*np.max(Age_vdb),  # small y-offset
             Names_vdb[i], fontsize=7, color='black', alpha=0.8)

#Add labels and titles for the plot
plt.xlabel('Age (Gyrs)')
plt.ylabel('Metalicity [Fe/H]')
plt.title('Age vs Metalicity (Van Den Berg)')

#Show plot
plt.show()


### =================== HARRIS 3D POSITION =================== ###

# 3D position 
plt.figure(1)
ax = plt.axes(projection='3d')
# Some data points black so they are visible on the plot
ax.scatter(X_Harris, Y_Harris, Z_Harris, c=Rotational_v_Harris, edgecolors='black', cmap='coolwarm')

# Create a circle (radius 10 kpc, centered at origin)
r = 10  # radius in kpc
theta = np.linspace(0, 2*np.pi, 200)
x_circle = r * np.cos(theta)
y_circle = r * np.sin(theta)
z_circle = np.zeros_like(theta)
ax.plot(x_circle, y_circle, z_circle, color='black', alpha=0.4, linewidth=2, label='10 kpc circle')
ax.plot_trisurf(x_circle, y_circle, z_circle, color='red', alpha=0.1, linewidth=0)

# Added Colour Bar to map heliocentric radial velocities
plt.colorbar(ax.collections[0], ax=ax, label = 'Heliocentric Radial Velocities (km/s)')
ax.set_xlabel('x Displacement (kpc)')
ax.set_ylabel('y Displacement (kpc)')
ax.set_zlabel('z Displacement (kpc)')
ax.legend

# Show plot
plt.show()

# Zoomed in plot closer to center to see circle more clearly

lim = 15.0 # axis limits for x, y, z

# Compute mask for points inside the 10 kpc circle in the x–y plane
dist_xy = np.sqrt(X_Harris**2 + Y_Harris**2)
inside_mask = dist_xy <= r

# Plotting zoomed in figure of previous graph
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_Harris, Y_Harris, Z_Harris,
                c=Rotational_v_Harris, edgecolors='black', cmap='coolwarm')

# Draw the 10 kpc circle in the x–y plane
theta = np.linspace(0, 2*np.pi, 300)
ax.plot(x_circle, y_circle, z_circle, color='red', alpha=0.7, linewidth=1.8)
ax.plot_trisurf(x_circle, y_circle, z_circle, color='red', alpha=0.1, linewidth=0)

# Label points inside the circle
for i in range(len(Names_Harris)):
    if inside_mask[i]:
        ax.text(X_Harris[i], Y_Harris[i], Z_Harris[i],
                str(Names_Harris[i]), color='black', fontsize=8, alpha=0.9)

# Axes labels and titles
ax.set_xlabel('x Displacement (kpc)')
ax.set_ylabel('y Displacement (kpc)')
ax.set_zlabel('z Displacement (kpc)')
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)
ax.set_title('Labels for Clusters Inside 10 kpc Circle')
plt.colorbar(sc, ax=ax, label='Heliocentric Radial Velocities (km/s)')

plt.show()

# 2d plot of x-y plan to more easily identify accreted clusters
ax.scatter(X_Harris, Y_Harris, s=30)
outside_mask = ~inside_mask
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(X_Harris, Y_Harris, s=30)

# Circle
circle = plt.Circle((0, 0), r, edgecolor='red', linewidth=1.6, facecolor='red', alpha=0.3, label='10 kpc circle')
ax.add_patch(circle)

# Label clusters outside of the circle
for i in range(len(Names_Harris)):
    if outside_mask[i]:
        ax.text(X_Harris[i], Y_Harris[i], str(Names_Harris[i]), fontsize=8, color='black', alpha=0.9)

# Axes labels, equal aspect, limits, colorbar
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