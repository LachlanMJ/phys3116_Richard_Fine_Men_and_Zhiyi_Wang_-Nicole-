import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# read Harris Part I Data
harris_part1 =pd.read_csv('HarrisPartI.csv')

# read Harris Part II Data
harris_part2 =pd.read_csv('HarrisPartIII.csv')

# read Krause 21 Data
Krause21 =pd.read_csv('Krause21.csv')

# read vandenBerg_Table2
vandenBerg_table2 =pd.read_csv('vandenBerg_table2.csv')

# Defining Variables for Krause Clusters
FeH_Krause = Krause21['FeH']
Age_Krause = Krause21['Age']
Name_Krause = Krause21['Object']

# scatter plot Krause 21 Data with colourmap
plt.scatter(Age_Krause, FeH_Krause, c = Age_Krause, cmap = 'coolwarm')
for i, txt in enumerate(Name_Krause):
    plt.annotate(txt, (Age_Krause[i], FeH_Krause[i]), fontsize=8)

#Add labels and titles for Krause Metalicity plot
plt.xlabel('Age of Krause Clusters')
plt.ylabel('Metalicity of Krause Clusters')
plt.title('Metalicity vs Age of Krause Clusters')

#Show plot
plt.show()

# Defining variables for Van Den Berg Clusters
FeH_vdb = vandenBerg_table2['FeH']
Age_vdb = vandenBerg_table2['Age']
Name_vdb = vandenBerg_table2['#NGC']

# scatter plot Van Der Berg Metalicity Data with colourmap
plt.scatter(Age_vdb, FeH_vdb, c = Age_vdb, cmap = 'coolwarm')
for i, txt in enumerate(Name_vdb):
    plt.annotate(txt, (Age_vdb[i], FeH_vdb[i]), fontsize=8)

#Add labels and titles for the plot
plt.xlabel('Age of Van Den Berg Clusters')
plt.ylabel('Metalicity of Van Den Berg Clusters')
plt.title('Metalicity vs Age of Van Den Berg Clusters')

# Defining Harris Variables
X_Harris = harris_part1['X']
Y_Harris = harris_part1['Y']
Z_Harris = harris_part1['Z']
v_r = harris_part2['v_r']
Name_Harris = harris_part1['ID']

#Show plot
plt.show()

plt.figure(1)
ax = plt.axes(projection='3d')
# Some data points black so they are visible on the plot
ax.scatter(X_Harris, Y_Harris, Z_Harris, c=v_r, edgecolors='black', cmap='coolwarm')
# Adding Labels
#for i in range(len(Name_Harris)):
#    ax.text(X_Harris[i], Y_Harris[i], Z_Harris[i], Name_Harris[i], color='black', fontsize=8)
# Added Colour Bar to map heliocentric radial velocities
plt.colorbar(ax.collections[0], ax=ax, label = 'Heliocentric Radial Velocities (km/s)')
ax.set_xlabel('x Displacement (kpc)')
ax.set_ylabel('y Displacement (kpc)')
ax.set_zlabel('z Displacement (kpc)')

# Show plot
plt.show()