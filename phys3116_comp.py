import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# read Harris Part I Data
harris_part1 =pd.read_csv(r"C:\Users\oscar\Downloads\HarrisPartI (1).csv")

# read Harris Part 2 Data
hariss_part2 =pd.read_csv(r"C:\Users\oscar\Downloads\HarrisPartIII.csv")

# read Krasue 21 Data
Krasue21 =pd.read_csv(r"C:\Users\oscar\Downloads\Krause21.csv")

# read vandenBerg_Table2
vanderberg_table2 =pd.read_csv(r"C:\Users\oscar\Downloads\vandenBerg_table2.csv")

# scatter plot Krasue 21 Data
plt.scatter(de['Age'], df['Fe'])

#Add labels and titles for the plot
plt.xlabel('Age')
plt.ylabel('fe(Iron)')
plt.title('Fe vs Age')

#Show plot
plt.show()
