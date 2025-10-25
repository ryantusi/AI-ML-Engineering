"""
Exploring Student Data

Imagine that you work for a school district and have collected some data on local students and their parents. You’ve been tasked with answering some important questions:

    How are students performing in their math classes?
    What do students’ parents do for work?
    How often are students absent from school?

In this project, you’ll explore and summarize some student data in order to answer these questions.

Data citation:

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science. http://archive.ics.uci.edu/ml/datasets/Student+Performance

Paulo Cortez, University of Minho, Guimarães, Portugal, http://www3.dsi.uminho.pt/pcortez
"""

# Load libraries
import pandas as pd
import numpy as np
import codecademylib3
import matplotlib.pyplot as plt
import seaborn as sns

# Import data
students = pd.read_csv('students.csv')

# Print first few rows of data
print(students.head())

# Print summary statistics for all columns
print(students.describe(include = 'all'))

# Calculate mean
print('mean math grade: ', students.math_grade.mean())

# Calculate median
print('median math grade: ', students.math_grade.median())

# Calculate mode
print('mode of math grades: ', students.math_grade.mode()[0])

# Calculate range
print('range of math grades: ', students.math_grade.max() - students.math_grade.min())

# Calculate standard deviation
print('SD of math grades: ', students.math_grade.std())

# Calculate MAD
print('MAD of math grades: ', students.math_grade.mad())

# Create a histogram of math grades
sns.histplot(x = 'math_grade', data = students)
plt.show()
plt.clf()

# Create a box plot of math grades
sns.boxplot(x = 'math_grade', data = students)
plt.show()
plt.clf()

# Calculate number of students with mothers in each job category
print(students.Mjob.value_counts())

# Calculate proportion of students with mothers in each job category
print(students.Mjob.value_counts(normalize=True))

# Create bar chart of Mjob
sns.countplot(x = 'Mjob', data = students)
plt.show()
plt.clf()

# Create pie chart of Mjob
students.Mjob.value_counts().plot.pie()
plt.show()
