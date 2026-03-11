# Iris Multiclass Classifier Project

## Project Overview
This project aims to build a multi-class classifier for the Iris flower dataset using various machine learning algorithms. The classifiers will be evaluated on their performance, and visualization will help understand the dataset and the outcomes.

## Data Loading
We will first load the Iris dataset using the Pandas library.

```python
import pandas as pd

# Load the Iris dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
                 header=None, 
                 names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
print(df.head())
```

## Data Preprocessing
Before training the models, we may need to preprocess the data:

1. **Handling Null Values**
2. **Encoding Categorical Variables**
3. **Feature Scaling**

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Check for null values
df.isnull().sum()

# Encode categorical target variable
df['class'] = LabelEncoder().fit_transform(df['class'])

# Feature and target variables
X = df.drop('class', axis=1)
y = df['class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## Model Training
We will use several models to classify the Iris dataset:

### Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
```

### Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
```

### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)
```

## Model Evaluation
Let's evaluate our models using accuracy and confusion matrix:
```python
from sklearn.metrics import accuracy_score, confusion_matrix

models = {'Logistic Regression': log_reg, 'Decision Tree': decision_tree, 'Random Forest': random_forest}

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f'Model: {name}')
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(confusion_matrix(y_test, y_pred))
    print('\n')
```

## Visualizations
We can visualize the decision boundaries and feature importance:

### Feature Importance for Random Forest
```python
import matplotlib.pyplot as plt
import numpy as np

# Get feature importance
importances = random_forest.feature_importances_
features = df.columns[:-1]
indices = np.argsort(importances)[::-1]

# Plot
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()
```

### Decision Boundary Visualization
```python
# Visualize decision boundaries for one feature (simplifying to 2D)
# Assuming sepal_length and sepal_width are used
from mlxtend.plotting import plot_decision_regions

X_2D = X_train[:, :2]  # Simplifying to first 2 features
log_reg.fit(X_2D, y_train)
plot_decision_regions(X_2D, y_train.values, clf=log_reg, legend=2)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Logistic Regression Decision Boundary')
plt.show()
```

## Conclusion
In this project, we explored various models to classify the Iris flower dataset. The performance metrics indicate that certain models outperformed others. Further tuning and exploration of additional models could provide better accuracy.