"""
Familiar: A Study In Data Analysis

Welcome to Familiar, a startup in the new market of blood transfusion! You’ve joined the team because you appreciate the flexible hours and extremely intelligent team, but the overeager doorman welcoming you into the office is a nice way to start your workday (well, work-evening).

Familiar has fallen into some tough times lately, so you’re hoping to help them make some insights about their product and help move the needle (so to speak).
"""

# Import libraries
import pandas as pd
import numpy as np

# Load datasets
lifespans = pd.read_csv('familiar_lifespan.csv')
iron = pd.read_csv('familiar_iron.csv')

# Print lifespan data
print(lifespans.head())

# Save lifespans for vein pack subscribers
vein_pack_lifespans = lifespans.lifespan[lifespans.pack=='vein']

# Calculate average lifespan for vein pack
print(np.mean(vein_pack_lifespans))

# Run one-sample t-test
from scipy.stats import ttest_1samp
tstat, pval = ttest_1samp(vein_pack_lifespans, 73)
print(pval)

# Save lifespans for artery pack subscribers
artery_pack_lifespans = lifespans.lifespan[lifespans.pack=='artery']

# Calculate artery pack life spans
print(np.mean(artery_pack_lifespans))

# Run two-sample t-test
from scipy.stats import ttest_ind
tstat, pval = ttest_ind(vein_pack_lifespans, artery_pack_lifespans)
print(pval)

# Inspect first 5 rows of iron dataset
print(iron.head())

# Create contingency table
Xtab = pd.crosstab(iron.pack, iron.iron)
print(Xtab)

# Run Chi-Square test
from scipy.stats import chi2_contingency
chi2, pval, dof, exp = chi2_contingency(Xtab)
print(pval)
