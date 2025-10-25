"""
FetchMaker

Congratulations! You’ve just started working at the hottest new tech startup, FetchMaker. FetchMaker’s mission is to match up prospective dog owners with their perfect pet. FetchMaker has been collecting data on their adoptable dogs, and it’s your job to analyze some of that data.
"""

# Import libraries
import numpy as np
import pandas as pd
import codecademylib3

# Import data
dogs = pd.read_csv('dog_data.csv')

# Inspect first few rows of data
print(dogs.head())

# Save the is_rescue column for whippets
whippet_rescue = dogs.is_rescue[dogs.breed == 'whippet']

# Calculate and print the number of whippet rescues
num_whippet_rescues = np.sum(whippet_rescue == 1)
print(num_whippet_rescues)

# Calculate and print the number of whippets
num_whippets = len(whippet_rescue)
print(num_whippets)

# Run a binomial test 
from scipy.stats import binom_test
pval = binom_test(num_whippet_rescues, num_whippets, .08)
print(pval)

# Save the weights of whippets, terriers, and pitbulls
wt_whippets = dogs.weight[dogs.breed == 'whippet']
wt_terriers = dogs.weight[dogs.breed == 'terrier']
wt_pitbulls = dogs.weight[dogs.breed == 'pitbull']

# Run an ANOVA 
from scipy.stats import f_oneway
Fstat, pval = f_oneway(wt_whippets, wt_terriers, wt_pitbulls)
print(pval)

# Subset to just whippets, terriers, and pitbulls
dogs_wtp = dogs[dogs.breed.isin(['whippet', 'terrier', 'pitbull'])]

# Run Tukey's Range Test
from statsmodels.stats.multicomp import pairwise_tukeyhsd
output = pairwise_tukeyhsd(dogs_wtp.weight, dogs_wtp.breed)
print(output)

# Subset to just poodles and shihtzus
dogs_ps = dogs[dogs.breed.isin(['poodle', 'shihtzu'])]

# Create a contingency table of color vs. breed
Xtab = pd.crosstab(dogs_ps.color, dogs_ps.breed)
print(Xtab)

# Run a Chi-Square Test
from scipy.stats import chi2_contingency
chi2, pval, dof, exp = chi2_contingency(Xtab)
print(pval)