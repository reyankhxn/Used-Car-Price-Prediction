import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

# Load Dataset
df = pd.read_csv("car_data.csv/car data.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# Remove Car Name column
df = df.drop("Car_Name", axis=1)

# Features and Target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Convert categorical columns
X = pd.get_dummies(X, drop_first=True)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
score = r2_score(y_test, predictions)

print("\nR² Score:", round(score, 2))

# Save Model
joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

print("\nModel Saved Successfully!")