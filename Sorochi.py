import matplotlib.pyplot as plt

def compute_sorori_shinzaemon(n_days=100):
    list_n_grains = [2**(day - 1) for day in range(1, n_days + 1)]
    list_total_grains = [sum(list_n_grains[:i + 1]) for i in range(n_days)]
    return list_n_grains, list_total_grains

def calculate_days_of_survival(total_grains, num_people, grains_per_day_per_person):
    return total_grains // (num_people * grains_per_day_per_person)

# Problem 1
list_n_grains_1, list_total_grains_1 = compute_sorori_shinzaemon(n_days=100)

# Problem 2 (Example: 10 days)
list_n_grains_2, list_total_grains_2 = compute_sorori_shinzaemon(n_days=10)

# Plotting
plt.plot(range(1, 101), list_n_grains_1)
plt.plot(range(1, 101), list_total_grains_1)
plt.plot(range(1, 11), list_n_grains_2)
plt.plot(range(1, 11), list_total_grains_2)

# Problem 3
total_grains_3 = list_total_grains_1[-1]
num_people_3, grains_per_day_per_person_3 = 5, 400
days_of_survival = calculate_days_of_survival(total_grains_3, num_people_3, grains_per_day_per_person_3)
print(f"With {total_grains_3} grains, {num_people_3} people can survive for {days_of_survival} days.")

plt.xlabel('Number of days')
plt.ylabel('Number of rice grains')
plt.title('Number of Rice Grains over Time')
plt.legend()
plt.show()
