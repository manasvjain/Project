import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('AIML/data/Housing.csv')

# Convert categorical data to numeric
df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop('price', axis=1)
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Print parameters
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Prediction
y_pred = model.predict(X_test)

# Accuracy (R² score)
print("Model Accuracy:", model.score(X_test, y_test))