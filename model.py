import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Dataset
data = {
    "Rooms": [2, 3, 4, 5, 3, 4, 6, 2, 5, 4],
    "Bathrooms": [1, 2, 3, 3, 2, 2, 4, 1, 3, 2],
    "Kitchens": [1, 1, 1, 2, 1, 1, 2, 1, 2, 1],
    "Washrooms": [1, 2, 2, 3, 2, 2, 4, 1, 3, 2],
    "Price": [2000000, 3500000, 5000000, 7000000,
              3200000, 4800000, 9000000, 1800000,
              7500000, 5200000]
}

df = pd.DataFrame(data)

X = df[["Rooms", "Bathrooms", "Kitchens", "Washrooms"]]
y = df["Price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully!")