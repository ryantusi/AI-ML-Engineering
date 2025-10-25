"""
Sampling Distributions Dance Party!

Get your dancing shoes ready! You are a DJ trying to make sure you are ready for a big party. You don’t have time to go through all the songs you can work with. Instead, you want to make sure that any sample of 30 songs from your playlist will get the party started. To do this, you will use the power of sampling distributions!

The dataset we are using for this project can be found here. For simplicity, we have removed some unnecessary columns.
"""

from helper_functions import choose_statistic, population_distribution, sampling_distribution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import codecademylib3

# task 1: load in the spotify dataset
spotify_data = pd.read_csv("spotify_data.csv")

# task 2: preview the dataset
print(spotify_data.head())

# task 3: select the relevant column
song_tempos = spotify_data['tempo']

# task 5: plot the population distribution with the mean labeled
population_distribution(song_tempos)

# task 6: sampling distribution of the sample mean
sampling_distribution(song_tempos, 30, "Mean")

# task 8: sampling distribution of the sample minimum
sampling_distribution(song_tempos, 30, "Minimum")

# task 10: sampling distribution of the sample variance
sampling_distribution(song_tempos, 30, "Variance")

# task 13: calculate the population mean and standard deviation
population_mean = np.mean(song_tempos)
population_std = np.std(song_tempos)

# task 14: calculate the standard error
standard_error = population_std/(30**.5)

# task 15: calculate the probability of observing an average tempo of 140bpm or lower from a sample of 30 songs
print(stats.norm.cdf(140, population_mean, standard_error))

# task 16: calculate the probability of observing an average tempo of 150bpm or higher from a sample of 30 songs
print(1-stats.norm.cdf(150, population_mean, standard_error))

# EXTRA
