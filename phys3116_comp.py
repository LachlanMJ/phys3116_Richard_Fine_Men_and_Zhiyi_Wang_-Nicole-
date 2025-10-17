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
Age_Krause = Krause21['Age']
FeH_Krause = Krause21['FeH']
Names = Krause21['Object']

# scatter plot Krause 21 Data
plt.scatter(Age_Krause, FeH_Krause)

#Add labels (names) next to each data point
for i in range(len(Krause21)):
    plt.text(Age_Krause[i] + 0.05 * np.max(Age_Krause) / len(Krause21),   #small x-offset
             FeH_Krause[i] + 0.05 * np.max(FeH_Krause) / len(Krause21),   # small y-offset
             Names[i], fontsize=7, color='darkred', alpha=0.8)

#Add labels and titles for the plot
plt.xlabel('Age of Krause Clusters')
plt.ylabel('Metalicity of Clusters')
plt.title('Age of Krause Clusters vs Metalicity')

#Show plot
plt.show()

# Defining variables for Van Den Berg Clusters
FeH_vdb = vandenBerg_table2['FeH']
Age_vdb = vandenBerg_table2['Age']
Names = vandenBerg_table2['Name']

# scatter plot Van Der Berg Data
plt.scatter(FeH_vdb, Age_vdb)

#Add labels (names) next to each data point
for i in range(len(vandenBerg_table2)):
    plt.text(FeH_vdb[i] + 0.02*np.max(FeH_vdb),   # small x-offset
             Age_vdb[i] + 0.02*np.max(Age_vdb),  # small y-offset
             Names[i], fontsize=7, color='darkred', alpha=0.8)

#Add labels and titles for the plot
plt.xlabel('Age of Van Den Berg Clusters')
plt.ylabel('Metalicity of Van Den Berg Clusters')
plt.title('Metalicity vs Age of Van Den Berg Clusters')

#Show plot
plt.show()

# Defining variables for Harris Part III Clusters
r_c = harris_part2['r_c']
sig_v = harris_part2['sig_v']
Names = harris_part2['ID']

# scatter plot Harris Part III Data
plt.scatter(r_c, sig_v)

#Add labels (names) next to each data point
for i in range(len(harris_part2)):
    plt.text(r_c[i] + 0.02*np.max(r_c),   # small x-offset
             sig_v[i] + 0.02*np.max(sig_v),  # small y-offset
             Names[i], fontsize=7, color='darkred', alpha=0.8)

#Add labels and titles for the plot
plt.xlabel('Core Radius (arcmin)')
plt.ylabel('Velocity Dispersion(km/s)')
plt.title('Core Radius vs Velocity Dispersion')

#Show plot
plt.show()

## Plot Stellar Mass vs. Age
#Define variables
Stellar_Mass = Krause21['Mstar']
Age = Krause21['Age']
Names = Krause21['AltName']   # Add cluster names

#Scatter plot Mass v.s. Age
plt.scatter(Stellar_Mass, Age)

#Add labels (names) next to each data point
for i in range(len(Krause21)):
    plt.text(Stellar_Mass[i] + 0.05 * np.max(Stellar_Mass) / len(Krause21),   #small x-offset
             Age[i] + 0.05 * np.max(Age) / len(Krause21),                     # small y-offset
             Names[i], fontsize=7, color='darkred', alpha=0.8)


#Add lables and titles for plot
plt.xlabel('Stellar Mass')
plt.ylabel('Stellar Age')
plt.title('Stellar Mass vs Stellar Age')

#Show plot
plt.show()


# Defining variables for Van Den Berg Clusters
FeH_vdb = vandenBerg_table2['FeH']
Age_vdb = vandenBerg_table2['R_G']
Names = vandenBerg_table2['Name']

# scatter plot Van Der Berg Data
plt.scatter(FeH_vdb, Age_vdb)

#Add labels (names) next to each data point
for i in range(len(vandenBerg_table2)):
    plt.text(FeH_vdb[i] + 0.02*np.max(FeH_vdb),   # small x-offset
             Age_vdb[i] + 0.02*np.max(Age_vdb),  # small y-offset
             Names[i], fontsize=7, color='darkred', alpha=0.8)

#Add labels and titles for the plot
plt.xlabel('Galactocentric Radius')
plt.ylabel('Metalicity of Van Den Berg Clusters')
plt.title('Metalicity vs Galactocentric Radius of Van Den Berg Clusters')

#Show plot
plt.show()

