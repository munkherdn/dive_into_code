import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "mtfuji_data.csv"
df = pd.read_csv(file_path)
print(df)

#[Problem 1] Data visualization
plt.figure(figsize=(10, 6))
plt.plot(df["x"], df["elevation"], linestyle='-')
plt.title("Mt. Fuji")
plt.xlabel("Position")
plt.ylabel("Elevation [m]")
plt.grid(True)
plt.show()


#[Problem 2] Create a function to calculate the gradient at a certain point
grad = df['elevation'].diff()/df['x'].diff()
df['gradient'] = grad

print('Points gradient:', grad)

#[Problem 3] Create a function to calculate the destination point

def get_destination(df, index, alpha):
    dest = alpha * df.loc[index, 'x'] * df.loc[index, 'gradient']
    return dest

print('Destination:', np.round(get_destination(df, index=136, alpha=0.2), 1))


#[Problem 4] Create a function that descends the mountain

def descents_mountain(df, index, alpha):
    dest = get_destination(df, index=136, alpha=0.2)
    descents = []
    for i in range(index, df.shape[0]):
        grad = df.loc[i, 'gradient']
        descents.append(dest-alpha*np.abs(grad))
        dest = descents[-1]

    return descents

dest = descents_mountain(df, index=136, alpha=0.2)

#[Problem 5] Visualization of the descent process
plt.plot(dest)
plt.show()

#[Problem 6] Changing the initial value
dest = descents_mountain(df, index=50, alpha=0.2)

plt.plot(dest)
plt.show()

#[Problem 7] Visualization of the descent process for each initial value
for i in range(df.shape[0]):
    dest = descents_mountain(df, index=i, alpha=0.2)
    plt.plot(dest)
plt.show()

#[Problem 8] (Advanced problem) Changing hyperparameters
for i in range(df.shape[0]):
    dest = descents_mountain(df, index=i, alpha=0.5)
    plt.plot(dest)
plt.show()
