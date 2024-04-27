import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# 0) Read data into a pandas dataframe
data = pd.read_csv('data_banknote_authentication.csv')

# 1) Pick the column named "class" as target variable y and all other columns as feature variables X
X = data.drop(columns=['class'])
y = data['class']

# 2) Split the data into training and testing sets with 80/20 ratio and random_state=20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# 3) Use support vector classifier with linear kernel to fit to the training data
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)

# 4) Predict on the testing data and compute the confusion matrix and classification report for linear kernel
y_pred_linear = svm_linear.predict(X_test)
conf_matrix_linear = confusion_matrix(y_test, y_pred_linear)
class_report_linear = classification_report(y_test, y_pred_linear)

# 5) Repeat steps 3 and 4 for the radial basis function kernel
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)
conf_matrix_rbf = confusion_matrix(y_test, y_pred_rbf)
class_report_rbf = classification_report(y_test, y_pred_rbf)

# 6) Compare the two SVM models in your own words
"""
The linear kernel SVM model tends to perform better when the data is linearly separable, 
while the radial basis function (RBF) kernel SVM model can capture more complex relationships 
and is often more effective in handling non-linear data. In this case, we can compare the accuracy,
precision, recall, and F1-score from the classification reports of both models to determine their relative performance.
"""

# Printing confusion matrix and classification reports
print("Linear Kernel SVM:")
print("Confusion Matrix:")
print(conf_matrix_linear)
print("\nClassification Report:")
print(class_report_linear)

print("\nRBF Kernel SVM:")
print("Confusion Matrix:")
print(conf_matrix_rbf)
print("\nClassification Report:")
print(class_report_rbf)
