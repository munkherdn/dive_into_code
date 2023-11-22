# Code to calculate the thickness when the paper is folded 43 times
THICKNESS = 0.00008

folded_thickness = THICKNESS * 2**43

print("Thickness: {} meters".format(folded_thickness))

print("Thickness: {:.2f} kilometers".format(folded_thickness / 1000))

THICKNESS = 0.00008
num_folds = 43

folded_thickness = THICKNESS

for _ in range(num_folds):
    folded_thickness *= 2

print("Thickness: {:.2f} kilometers".format(folded_thickness / 1000))

import time
start = time.time()
folded_thickness = THICKNESS * 2**43
elapsed_time = time.time() - start
print("Using exponentiation: time {}[s]".format(elapsed_time))

start = time.time()
folded_thickness = THICKNESS
for _ in range(43):
    folded_thickness *= 2
elapsed_time = time.time() - start
print("Using for statement: time {}[s]".format(elapsed_time))

THICKNESS = 0.00008
num_folds = 43

folded_thickness_list = [THICKNESS]

for _ in range(num_folds):
    folded_thickness *= 2
    folded_thickness_list.append(folded_thickness)

print("Number of elements in the list: {}".format(len(folded_thickness_list)))

import matplotlib.pyplot as plt

# Display the graph
plt.title("Thickness of Folded Paper")
plt.xlabel("Number of Folds")
plt.ylabel("Thickness [m]")
plt.plot(folded_thickness_list)
plt.show()

# Display a customized line graph
plt.title("Thickness of Folded Paper")
plt.xlabel("Number of Folds")
plt.ylabel("Thickness [m]")
plt.plot(folded_thickness_list, color='purple', linewidth=4, linestyle='--', marker='o', markersize=5)
plt.show()
