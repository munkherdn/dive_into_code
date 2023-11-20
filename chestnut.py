import matplotlib.pyplot as plt

def calculate_time_to_cover(target_volume, growth_rate, interval):
    time_list = [0]  # Start at time 0
    volume_list = [1]  # Initial volume of one chestnut bun

    current_volume = 1
    time = 0

    while current_volume < target_volume:
        time += interval
        current_volume *= 2  # Byevine doubles the volume

        time_list.append(time)
        volume_list.append(current_volume)

    return time_list, volume_list

def plot_growth_curve(time_list, volume_list):
    """
    Plot the growth curve.

    Parameters:
    - time_list: List of time points
    - volume_list: List of volumes corresponding to each time point
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time_list, volume_list, marker='o', linestyle='-', color='b')
    plt.title('Chestnut Bun Growth Over Time')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Chestnut Bun Volume')
    plt.grid(True)
    plt.show()

solar_system_volume = 4.1e67
growth_rate = 2
sprinkle_interval = 5

time_points, volume_points = calculate_time_to_cover(solar_system_volume, growth_rate, sprinkle_interval)

plot_growth_curve(time_points, volume_points)
