# Problem 1: Generating random numbers
import numpy as np

np.random.seed(0)

# Mean vector
mean = np.array([-3, 0])

# Covariance matrix
cov = np.array([[1.0, 0.8], [0.8, 1.0]])

# Generate 500 random numbers
data1 = np.random.multivariate_normal(mean, cov, 500)

# Problem 2: Scatter plot visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(data1[:, 0], data1[:, 1], marker='o', color='blue', alpha=0.7, label='Problem 1')
plt.title('Scatter plot')
plt.legend()
plt.show()

# Problem 3: Histogram visualization
plt.figure(figsize=(12, 6))

# Histogram for dimension 1
plt.subplot(1, 2, 1)
plt.hist(data1[:, 0], bins=20, range=(-6, 3), color='blue', alpha=0.7)
plt.xlabel('Dimension 1')
plt.ylabel('Frequency')
plt.title('Histogram of Dimension 1')

# Histogram for dimension 2
plt.subplot(1, 2, 2)
plt.hist(data1[:, 1], bins=20, range=(-6, 3), color='blue', alpha=0.7)
plt.xlabel('Dimension 2')
plt.ylabel('Frequency')
plt.title('Histogram of Dimension 2')

plt.subplots_adjust(left=0.15, wspace=0.3)
plt.show()

# Problem 4: Addition of data
mean2 = np.array([0, -3])
data2 = np.random.multivariate_normal(mean2, cov, 500)

# Problem 4: Visualization
plt.figure(figsize=(8, 6))
plt.scatter(data1[:, 0], data1[:, 1], marker='o', color='blue', alpha=0.7, label='Problem 1')
plt.scatter(data2[:, 0], data2[:, 1], marker='o', color='red', alpha=0.7, label='Problem 4')
plt.title('Scatter plot')
plt.legend()
plt.show()

# Problem 5: Data combination
data = np.vstack((data1, data2))


# Problem 6: Labeling
labels = np.zeros((1000, 1))
labels[500:] = 1


combined_data = np.hstack((data, labels))


print(combined_data.shape)
