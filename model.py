import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Sample data provided by the user
data = {
    'BHK': [3, 2, 4, 3, 3, 3, 4, 4, 3, 2],
    'TYPE': ['Apartment', 'Apartment', 'Villa', 'Apartment', 'Apartment', 'Apartment', 'Apartment', 'Independent House', 'Apartment', 'Apartment'],
    'AREA': ['Thaltej', 'Bopal', 'Sanand', 'Bopal', 'Gota', 'Memnagar', 'Vastrapur', 'Chandkheda', 'Shela', 'Nikol'],
    'C_STATUS': ['Ready to move', 'Under Construction', 'Under Construction', 'Under Construction', 'Under Construction', 'Under Construction', 'Ready to move', 'Ready to move', 'Under Construction', 'Ready to move'],
    'SQ_FT': [1385, 1250, 1536, 1400, 2034, 1672, 2727, 4095, 1436, 1215],
    'CITY': ['Ahmedabad', 'Ahmedabad', 'Ahmedabad', 'Ahmedabad', 'Ahmedabad', 'Ahmedabad', 'Ahmedabad', 'Ahmedabad', 'Ahmedabad', 'Ahmedabad'],
    'PRICE': [4800, 3956, 5880, 2636, 3650, 6100, 6200, 4517, 3043, 2592]
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Define the features and target variable
X = df[['BHK', 'SQ_FT']]
y = df['PRICE']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model as a .pkl file
joblib.dump(model, 'house_price_prediction_model.pkl')
