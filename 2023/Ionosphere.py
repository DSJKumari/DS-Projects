import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Fetching data from UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
column_names = [f'feature_{i}' for i in range(1, 35)] + ['label']
ionosphere_data = pd.read_csv(url, header=None, names=column_names)

# Convert 'g' and 'b' labels to 1 and 0
ionosphere_data['label'] = ionosphere_data['label'].map({'g': 1, 'b': 0})

# Split the data into features and labels
X = ionosphere_data.drop('label', axis=1)
y = ionosphere_data['label']

# Split the data into training and testing dataets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# K Neighbours Classifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create a KNN model
knn_model = KNeighborsClassifier()

# Train the model
knn_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_knn = knn_model.predict(X_test_scaled)

# Evaluate the model
accuracy_knn = accuracy_score(y_test, y_pred_knn)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)

# Confusion Matrix
print("\nKNN Confusion Matrix:")
print(conf_matrix_knn)
# Percentage of Accuracy
print(f"KNN Accuracy: {accuracy_knn * 100:.2f}%")

#---------------------------------------------------------------------------------------------
# Logistic Regression

# Create a logistic regression model
logreg_model = LogisticRegression(random_state=42)

# Training the model
logreg_model.fit(X_train_scaled, y_train)

# Classification on the Test Dataset
y_pred = logreg_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
Con_mat = confusion_matrix(y_test, y_pred)

# Confusion Matrix
print("\nLogisticRegression Confusion Matrix:")
print(Con_mat)
# Accuracy Percentage
print(f"Accuracy: {accuracy* 100:.2f}%")

#---------------------------------------------------------------------------------------------
# SVM

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create an SVM model
svm_model = SVC(random_state=42)

# Train the model
svm_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_svm = svm_model.predict(X_test_scaled)

# Evaluate the model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
classification_rep_svm = classification_report(y_test, y_pred_svm)

# Confusion Matrix
print("\nSVM Confusion Matrix:")
print(conf_matrix_svm)
# Accuracy Percentage
print(f"SVM Accuracy: {accuracy_svm * 100:.2f}%")

#----------------------------------------------------------------------------------------------

