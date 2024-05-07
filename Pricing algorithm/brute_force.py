import time
import numpy as np
import itertools
import matplotlib.pyplot as plt

# Define the range of possible values for each position in the array
values = range(11)

# Define the lengths of the arrays
lengths = range(1, 8)

# Initialize a list to store the elapsed times
times = []

# For each length
for length in lengths:
    # Start the timer
    start_time = time.time()

    # Generate all combinations
    combinations = list(itertools.product(values, repeat=length))

    # Stop the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Add the elapsed time to the list
    times.append(elapsed_time)

# Plot the elapsed times
#plt.plot(lengths, times, marker='o')
#plt.xlabel('Array Length')
#plt.ylabel('Elapsed Time (seconds)')
#plt.title('Time to Generate All Combinations')
#plt.grid(True)
#plt.show()


# Define the range of possible values for each position in the array
k = 11

# Define the lengths of the arrays
n_values = range(1, 25)

# Given that size 7 takes about 1 second, estimate the constant factor
# We know that time = constant * k^n, so constant = time / (k^n)
constant = 1 / (k ** 7)

# Calculate the theoretical times
theoretical_times = [constant * k ** n for n in n_values]

# Calculate the theoretical times in days
theoretical_times_days = [t / 86400 for t in theoretical_times]

# Calculate the theoretical times in years
theoretical_times_years = [t / 365.25 for t in theoretical_times_days]

# Plot the theoretical times in years
plt.plot(n_values, theoretical_times_years, marker='o')
plt.xlabel('Array Length')
plt.ylabel('Theoretical Time (years)')
plt.title('Theoretical Time to Generate All Combinations')
plt.grid(True)
plt.yscale('log')
plt.show()