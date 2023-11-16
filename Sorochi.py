import matplotlib.pyplot as plt

def calculate_rice_grains(day):
    return 2 ** (day - 1)

def calculate_total_rice_grains(day):
    total_grains = sum(calculate_rice_grains(i) for i in range(1, day + 1))
    return total_grains

# Calculate the number of grains on each day and the total grains
days = list(range(1, 101))
rice_per_day = [calculate_rice_grains(day) for day in days]
total_rice_per_day = [calculate_total_rice_grains(day) for day in days]

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(days, rice_per_day, label='Rice grains on the day', marker='o')
plt.plot(days, total_rice_per_day, label='Total rice grains by the day', marker='o')
plt.xlabel('Days')
plt.ylabel('Number of Rice Grains')
plt.title('Change in Number of Rice Grains Over 100 Days')
plt.legend()
plt.grid(True)
plt.show()

# Output the total number of rice grains on the 100th day
day_100_total_rice = calculate_total_rice_grains(100)
print(f'Total number of rice grains on the 100th day: {day_100_total_rice}')
