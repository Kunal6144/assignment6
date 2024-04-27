import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Step 0: Read data into a pandas dataframe
data = pd.read_csv('weight-height.csv')

# Step 1: Pick the target variable y as weight in kilograms, and the feature variable X as height in centimeters
X = data['Height'].values.reshape(-1, 1)  # Reshape to ensure it's a 2D array
y = data['Weight']

# Step 2: Split the data into training and testing sets with 80/20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Scale the training and testing data using normalization and standardization
scaler_norm = MinMaxScaler()
X_train_norm = scaler_norm.fit_transform(X_train)
X_test_norm = scaler_norm.transform(X_test)

scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)

# Step 4: Fit a KNN regression model with k=5 to the training data without scaling,
# predict on unscaled testing data and compute the R2 value
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_unscaled = knn_model.predict(X_test)
r2_unscaled = r2_score(y_test, y_pred_unscaled)

# Step 5: Repeat step 4 for normalized data
knn_model.fit(X_train_norm, y_train)
y_pred_norm = knn_model.predict(X_test_norm)
r2_norm = r2_score(y_test, y_pred_norm)

# Step 6: Repeat step 4 for standardized data
knn_model.fit(X_train_std, y_train)
y_pred_std = knn_model.predict(X_test_std)
r2_std = r2_score(y_test, y_pred_std)

# Step 7: Compare the models in terms of their R2 value
print("R2 Score without scaling:", r2_unscaled)
print("R2 Score with normalization:", r2_norm)
print("R2 Score with standardization:", r2_std)
