import gc
import joblib
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from mlxtend.plotting import plot_decision_regions

warnings.filterwarnings('ignore')

iris = load_iris()
target_names = {0:'setosa',
                1:'versicolor',
                2:'virginica'}

X = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width',
                                     'petal_length', 'petal_width'])
y = pd.DataFrame(iris.target, columns=['Species'])


df = pd.concat([X, y], axis=1)

#[Assignment 1] Select features and categories for practice
# Scatter plots
df['Species'] = df['Species'].map(target_names)
df = df.loc[df['Species'].isin(['versicolor', 'virginica']), ['sepal_length',
                                                              'petal_length',
                                                              'Species']]
df = df.reset_index(drop=True)

sns.scatterplot(x='sepal_length', y='petal_length', hue='Species', data=df)
plt.title('Scatter Plot of Sepal Length vs Sepal Width')
plt.show()


#[Question 2] Data analysis

# Box plot
sns.boxplot(x='Species', y='sepal_length', data=df)
plt.title('Distribution of Sepal Length for Each Label')
plt.show()

# Violin plot
sns.violinplot(x='Species', y='sepal_length', data=df)
plt.title('Distribution of Sepal Length for Each Label')
plt.show()


sns.pairplot(df, hue='Species')
plt.suptitle('Scatterplot Matrix', y=1.02)
plt.show()

# Correlation coefficient matrix
target_names = {'versicolor':0,
                'virginica':1}
df['Species'] = df['Species'].map(target_names)
correlation_matrix = df.corr()

# Heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Coefficient Matrix')
plt.show()


#[Problem 3] Division of preprocessing/training data and verification data
train_X, test_X, train_Y, test_y = train_test_split(df[['sepal_length', 'petal_length']], df['Species'], test_size=0.25,
                                                     shuffle=True, random_state=42)

print("Train data: \n x = {} \n y = {} \n\n Test data: \n x = {} \n y = {}"
      "\n".format(train_X, test_X, train_Y, test_y))

#[Problem 4] Pretreatment/Standardization
scaler = StandardScaler()
scaler.fit(train_X)
x_train_scaled = scaler.transform(train_X)
x_test_scaled = scaler.transform(test_X)

#[Problem 5] Learning and estimation
def accuracy(predicted, true):
    count = 0
    n = len(predicted)
    for i in range(n):
        if (predicted[i] != true[i]):
            count+=1
    return 1-(count/n)

def Kneighbors(data, label, test, n):
    neigh = KNeighborsClassifier(n_neighbors=n)
    neigh.fit(data, label.values.ravel())
    predicted = neigh.predict(test)
    true = test_y.values

    print("Predicted labels: {}".format(predicted))
    print("True labels: {}".format(true))
    print("Accuracy {}-nn: {:4f}".format(n, accuracy(predicted, true)))

Kneighbors(x_train_scaled, train_Y, x_test_scaled, 20)
Kneighbors(x_train_scaled, train_Y, x_test_scaled, 5)
Kneighbors(x_train_scaled, train_Y, x_test_scaled, 3)


#[Problem 6] Evaluation
neigh = KNeighborsClassifier(n_neighbors=20)
neigh.fit(x_train_scaled, train_Y.values.ravel())
predicted = neigh.predict(x_test_scaled)

print('Accuracy:', accuracy_score(test_y, predicted))
print('Precision:', precision_score(test_y, predicted))
print('Recall:', recall_score(test_y, predicted))
print('F1 score:', f1_score(test_y, predicted))
print('Confusion matrix:', confusion_matrix(test_y, predicted))

#[Problem 7] Visualization
def decision_region(X, y, model, step=0.01, title='decision region', xlabel='xlabel', ylabel='ylabel', target_names=['versicolor', 'virginica']):

    # setting
    scatter_color = ['red', 'blue']
    contourf_color = ['pink', 'skyblue']
    n_class = 2
    # pred
    mesh_f0, mesh_f1  = np.meshgrid(np.arange(np.min(X[:,0])-0.5, np.max(X[:,0])+0.5, step), np.arange(np.min(X[:,1])-0.5, np.max(X[:,1])+0.5, step))
    mesh = np.c_[np.ravel(mesh_f0),np.ravel(mesh_f1)]
    y_pred = model.predict(mesh).reshape(mesh_f0.shape)
    # plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.contourf(mesh_f0, mesh_f1, y_pred, n_class-1, cmap=ListedColormap(contourf_color))
    plt.contour(mesh_f0, mesh_f1, y_pred, n_class-1, colors='y', linewidths=3, alpha=0.5)
    for i, target in enumerate(set(y)):
        plt.scatter(X[y==target][:, 0], X[y==target][:, 1], s=80, color=scatter_color[i], label=target_names[i], marker='o')
    patches = [mpatches.Patch(color=scatter_color[i], label=target_names[i]) for i in range(n_class)]
    plt.legend(handles=patches)
    plt.legend()
    plt.show()

#knn
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train_scaled, train_Y.values.ravel())
neigh_predicted = neigh.predict(x_test_scaled)
decision_region(x_train_scaled, train_Y.values.ravel(), neigh)


#[Problem 8] Learning by other methods
#knn
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train_scaled, train_Y.values.ravel())
neigh_predicted = neigh.predict(x_test_scaled)
decision_region(x_train_scaled, train_Y.values.ravel(), neigh)
print("Accuracy of standardized Kneighbor: {}".format(accuracy_score(test_y, neigh_predicted)))



#logistic
logistics = LogisticRegression(random_state=0).fit(x_train_scaled, train_Y.values.ravel())
logistics_predicted = logistics.predict(x_test_scaled)
decision_region(x_test_scaled, test_y.values.ravel(), logistics)
print("Accuracy of standardized Logistics Regression: {}".format(accuracy_score(test_y, logistics_predicted)))

##svm
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
svc_model = clf.fit(x_train_scaled, train_Y.values.ravel())
svc_predicted = svc_model.predict(x_test_scaled)
decision_region(x_test_scaled, test_y.values.ravel(), svc_model)
print("Accuracy of standardized SVM: {}".format(accuracy_score(test_y, svc_predicted)))

#Decision Tree
clf = DecisionTreeClassifier(random_state=0)
decision_tree_predict = cross_val_score(clf, x_train_scaled, train_Y.values.ravel(), cv=10)
print('Cross val score:', decision_tree_predict)

#Random Forest
clf = RandomForestClassifier(max_depth=2, random_state=0)
random_forest_model = clf.fit(x_train_scaled, train_Y.values.ravel())
random_forest_predicted = random_forest_model.predict(x_test_scaled)
decision_region(x_test_scaled, test_y.values.ravel(), random_forest_model)

print("Accuracy of standardized Random Forest: {}".format(accuracy_score(test_y,
                                                                         random_forest_predicted)))
##merge
name = ["Kneighbor", "LogisticsRegression", "SVC", "RandomForest"]
predicts = [neigh_predicted, logistics_predicted, svc_predicted, random_forest_predicted]

test_standart = []
for i in range(len(name)):
    test_standart.append(predicts[i])

#[Problem 9] (Advanced task) Comparison with and without standardization
train_X, test_X, train_Y, test_y = train_test_split(df[['sepal_length', 'petal_length']], df['Species'], test_size=0.25,
                                                     shuffle=True, random_state=42)
## knn
new_neigh = KNeighborsClassifier(n_neighbors=3)
new_neigh.fit(train_X.values, train_Y.values.ravel())
new_neigh_predicted = new_neigh.predict(test_X.values)
decision_region(test_X.values, test_y.values.ravel(), new_neigh)

print("Accuracy of non-standardized Kneighbor: {}".format(accuracy_score(test_y, new_neigh_predicted)))

#logistic
new_logistics = LogisticRegression(random_state=0).fit(train_X.values, train_Y.values.ravel())
new_logistics_predicted = new_logistics.predict(test_X.values)
decision_region(test_X.values, test_y.values.ravel(), new_logistics)
print("Accuracy of non-standardized Logistics Regression: {}".format(accuracy_score(test_y, new_logistics_predicted)))

## SVM
new_svc_model = SVC.fit(train_X.values, train_Y.values.ravel())
new_svc_predicted = new_svc_model.predict(test_X.values)
decision_region(test_X.values, test_y.values.ravel(), new_svc_model)
print("Accuracy of non-standardized SVC {}".format(accuracy_score(test_y, new_svc_predicted)))

## Random forest
new_clf = RandomForestClassifier(max_depth=2, random_state=0)
new_random_forest_model = new_clf.fit(train_X.values, train_Y.values.ravel())
new_random_forest_predicted = new_random_forest_model.predict(test_X.values)
decision_region(test_X.values, test_y.values.ravel(), new_random_forest_model)
print("Accuracy of non-standardized Random Forest: {}".format(accuracy_score(test_y,
                                                                             new_random_forest_predicted)))
new_name = ["Kneighbor", "LogisticsRegression", "SVC", "RandomForest"]
new_predicts = [new_neigh_predicted, new_logistics_predicted, new_svc_predicted, new_random_forest_predicted]

test_y_no_stand = []
for i in range(len(new_name)):
    test_y_no_stand.append(new_predicts[i])

for name, pred_stand, pred_no_stand in zip(new_name,
                                           test_standart,
                                           test_y_no_stand):
    print(f'Standartized {name} accuracy:', accuracy(pred_stand, test_y))
    print(f'Not {name} accuracy:', accuracy(pred_no_stand, test_y))

#[Problem 10] Multi-classes
X = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width',
                                     'petal_length', 'petal_width'])
y = pd.DataFrame(iris.target, columns=['Species'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

scaler = StandardScaler()

scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train_scaled, y_train.values.ravel())
neigh_predicted = neigh.predict(x_test_scaled)
plot_decision_regions(x_test_scaled, y_test.values.ravel(), clf=neigh, legend=2)
print("Accuracy of standardized KNeighbor: {}".format(accuracy_score(y_test, neigh_predicted)))

## logistic regression
new_logistics = LogisticRegression(random_state=0).fit(x_train_scaled, y_train.values.ravel())
new_logistics_predicted = new_logistics.predict(x_test_scaled)
plot_decision_regions(x_test_scaled, y_test.values.ravel(), clf=new_logistics, legend=2)

print("Accuracy of standardized Logistic Regression: {}".format(accuracy_score(y_test, new_logistics_predicted)))

#SVM
new_clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
new_svc_model = new_clf.fit(x_train_scaled, y_train.values.ravel())
new_svc_predicted = new_svc_model.predict(x_test_scaled)
plot_decision_regions(x_test_scaled, y_test.values.ravel(), clf=new_svc_model, legend=2)
print("Accuracy of standardized SVC: {}".format(accuracy_score(y_test, new_svc_predicted)))

## Random Forest
new_clf = RandomForestClassifier(max_depth=2, random_state=0)
new_random_forest_model = new_clf.fit(x_train_scaled, y_train.values.ravel())
new_random_forest_predicted = new_random_forest_model.predict(x_test_scaled)
plot_decision_regions(x_test_scaled, y_test.values.ravel(), clf=new_random_forest_model, legend=2)
print("Accuracy of standardized Random Forest: {}".format(accuracy_score(y_test, new_random_forest_predicted)))
