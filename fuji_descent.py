import pandas as pd
import matplotlib.pyplot as plt

file_path = "mtfuji_data.csv"
df = pd.read_csv(file_path)

plt.figure(figsize=(10, 6))
plt.plot(df["x"], df["elevation"], linestyle='-')
plt.title("Mt. Fuji")
plt.xlabel("Position")
plt.ylabel("Elevation [m]")
plt.grid(True)
plt.show()
