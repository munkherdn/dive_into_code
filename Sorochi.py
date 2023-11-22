import matplotlib.pyplot as plt

def compute_sorori_shinzaemon(n_days=100):

    list_n_grains = []  # List to store the number of grains received each day
    list_total_grains = []  # List to store the total number of grains received by a certain day
    total_grains = 0  # Variable to keep track of the total number of grains

    for day in range(1, n_days + 1):
        grains_on_current_day = 2 ** (day - 1)
        total_grains += grains_on_current_day

        list_n_grains.append(grains_on_current_day)
        list_total_grains.append(total_grains)

    return list_n_grains, list_total_grains

# Problem 1
list_n_grains_1, list_total_grains_1 = compute_sorori_shinzaemon(n_days=100)

# Problem 2 (Example: 10 days)
list_n_grains_2, list_total_grains_2 = compute_sorori_shinzaemon(n_days=10)

# Plotting
plt.plot(range(1, 101), list_n_grains_1, label='Grains on the nth day (Problem 1)')
plt.plot(range(1, 101), list_total_grains_1, label='Total grains by nth day (Problem 1)')
plt.plot(range(1, 11), list_n_grains_2, label='Grains on the nth day (Problem 2)')
plt.plot(range(1, 11), list_total_grains_2, label='Total grains by nth day (Problem 2)')

plt.xlabel('Number of days')
plt.ylabel('Number of rice grains')
plt.title('Number of Rice Grains over Time')
plt.legend()
plt.show()
