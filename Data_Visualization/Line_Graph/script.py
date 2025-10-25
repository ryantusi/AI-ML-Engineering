"""
Sublime Limes' Line Graphs

In this project, you will be acting as a data analyst for an online lime retailer called Sublime Limes. People all over the world can use this product to get the freshest, best-quality limes delivered to their door. One of the product managers at Sublime Limes would like to gain insight into the customers and their sales habits. Using Matplotlib, you’ll create some different line graphs (or maybe, lime graphs) to communicate this information effectively.
"""

# Sublime Limes Line Graphs Project
import codecademylib
from matplotlib import pyplot as plt

# Months of the year
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Total site visits per month
visits_per_month = [9695, 7909, 10831, 12942, 12495, 16794, 
                    14161, 12762, 12777, 12439, 10309, 8724]

# Lime species sold each month
key_limes_per_month = [92.0, 109.0, 124.0, 70.0, 101.0, 79.0, 
                       106.0, 101.0, 103.0, 90.0, 102.0, 106.0]
persian_limes_per_month = [67.0, 51.0, 57.0, 54.0, 83.0, 90.0, 
                           52.0, 63.0, 51.0, 44.0, 64.0, 78.0]
blood_limes_per_month = [75.0, 75.0, 76.0, 71.0, 74.0, 77.0, 
                         69.0, 80.0, 63.0, 69.0, 73.0, 82.0]


# --- Step 3: Create Figure ---
plt.figure(figsize=(12, 8))

# --- Step 4-5: Create Subplots ---
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

# --- Page Visits Plot ---
x_values = range(len(months))
ax1.plot(x_values, visits_per_month, marker='o', color='limegreen')

# Label axes
ax1.set_xlabel("Month")
ax1.set_ylabel("Total Page Visits")

# Set ticks and tick labels
ax1.set_xticks(x_values)
ax1.set_xticklabels(months)

# Title for subplot 1
ax1.set_title("Total Page Visits per Month")

# --- Lime Species Plot ---
ax2.plot(x_values, key_limes_per_month, color='green', marker='o', label='Key Limes')
ax2.plot(x_values, persian_limes_per_month, color='orange', marker='o', label='Persian Limes')
ax2.plot(x_values, blood_limes_per_month, color='red', marker='o', label='Blood Limes')

# Label axes and ticks
ax2.set_xlabel("Month")
ax2.set_ylabel("Limes Sold")
ax2.set_xticks(x_values)
ax2.set_xticklabels(months)

# Add legend and title
ax2.legend()
ax2.set_title("Limes Sold per Month by Species")

# --- Adjust layout and Save ---
plt.tight_layout()
plt.savefig("sublime_limes_sales_and_visits.png")

plt.show()
