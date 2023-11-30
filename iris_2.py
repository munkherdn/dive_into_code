#problem 1
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

X = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

y = pd.DataFrame(iris.target, columns=['Species'])

#problem 2
df = pd.concat([X, y], axis=1)

#problem 3
print(df.head(4))

print(df['Species'].value_counts())
print(df.isnull().sum())
print(df.describe())

#problem 4
sepal_width1 = df['sepal_width']
sepal_width2 = df.loc[:, 'sepal_width']

data_50_99 = df.iloc[50:100]

petal_length_50_99 = df.loc[50:99, 'petal_length']

petal_width_0_2 = df[df['petal_width'] == 0.2]

#problem 5

import matplotlib.pyplot as plt
import seaborn as sns

# Pie chart
plt.pie(df['Species'].value_counts(), labels=iris.target_names, autopct='%1.1f%%')
plt.title('Number of Samples per Label')
plt.show()

# Box plot
sns.boxplot(x='Species', y='sepal_length', data=df)
plt.title('Distribution of Sepal Length for Each Label')
plt.show()

# Violin plot
sns.violinplot(x='Species', y='sepal_length', data=df)
plt.title('Distribution of Sepal Length for Each Label')
plt.show()
