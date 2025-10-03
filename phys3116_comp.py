import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# read Harris Part I Data
harris_part1 =pd.read_csv('HarrisPartI.csv')

# read Harris Part 2 Data
harris_part2 =pd.read_csv('HarrisPartIII.csv')

# read Krause 21 Data
Krause21 =pd.read_csv('Krause21.csv')

# read vandenBerg_Table2
vandenBerg_table2 =pd.read_csv('vandenBerg_table2.csv')

# Defining Variables for Krause Clusters
FeH_Krause = Krause21['FeH']
Age_Krause = Krause21['Age']

# scatter plot Krause 21 Data
plt.scatter(Age_Krause, FeH_Krause)

#Add labels and titles for the plot
plt.xlabel('Age of Krause Clusters')
plt.ylabel('Metalicity of Krause Clusters')
plt.title('Metalicity vs Age of Krause Clusters')

#Show plot
plt.show()

# Defining variables for Van Den Berg Clusters
FeH_vdb = vandenBerg_table2['FeH']
Age_vdb = vandenBerg_table2['Age']

# scatter plot Van Der Berg Data
plt.scatter(FeH_vdb, Age_vdb)

#Add labels and titles for the plot
plt.xlabel('Age of Van Den Berg Clusters')
plt.ylabel('Metalicity of Van Den Berg Clusters')
plt.title('Metalicity vs Age of Van Den Berg Clusters')

#Show plot
plt.show()

# Defining variables for Van Den Berg Clusters
r_c = harris_part2['r_c']
sig_v = harris_part2['sig_v']

# scatter plot Van Der Berg Data
plt.scatter(r_c, sig_v)

#Add labels and titles for the plot
plt.xlabel('Core Radius (arcmin)')
plt.ylabel('Velocity Dispersion(km/s)')
plt.title('Core Radius vs Velocity Dispersion')

#Show plot
plt.show()