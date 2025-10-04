# Week 2 Meeting

Meeting Started at 1:10pm, 26/09/2025

Atendees: Lachlan, Oscar, Nicole

- We are doing Option 1: Globular Clusters in the Milky Way
- We have to analyse a lot of data and look at the metalicity and age of these globular clusters, allowing us to draw conclusions, connections, relationships
- Plotting the data and checking to see if the plots follow known relationships
- This week, we will all download the CSV files and get python to read the CSV files

# PHYS3116 Group Project – Week 3 Meeting Notes

**Date:** 03/10/2025  
**Time:** 16:00 – 16:10  
**Attendees:** Nicole, Oscar, Lachlan 

## Agenda
- Follow up from Week 2: loading datasets and making initial plots.  
- Add new Mass vs Age plot for Krause clusters.  
- Confirm Python code works and plots are readable.  

## Work Completed
- Loaded datasets:
  - Harris Catalogue (Part I and III)  
  - Krause et al. (2021)  
  - VandenBerg et al. (2013)  
- Python libraries imported: `numpy`, `pandas`, `matplotlib.pyplot`.  
- Scatter plots produced:
  1. **Age vs [Fe/H]** for Krause (2021).  
  2. **Age vs [Fe/H]** for VandenBerg (2013).  
  3. **Core radius vs velocity dispersion** from Harris (2010).  
  4. **Stellar mass vs age** for Krause clusters (`Mstar` vs `Age`).  

## Observations
- Most clusters follow the expected age–metallicity trend.  
- Mass vs age plot shows that older clusters tend to be more massive, consistent with survivability selection effects.  
- Code is now modular and readable, with appropriate labels and titles on all plots.  

## Possible Next Steps
- Investigate outliers in the Age–[Fe/H] relation.  
- Consider adding kinematic data (radial velocities, proper motions) to explore potential accreted clusters.  
- Merge datasets where possible and start simple classification of clusters (in-situ vs accreted).  
