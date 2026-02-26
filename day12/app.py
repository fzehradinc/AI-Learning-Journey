import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import streamlit as st
import numpy as np

# Read data
try:
    df = pd.read_excel('cars.xls')
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Define features, making sure columns exist
expected_features = ['Mileage', 'Cylinder', 'Liter', 'Doors', 'Make', 'Model', 'Trim', 'Type', 'Cruise', 'Sound', 'Leather']
features = [col for col in expected_features if col in df.columns]

x = df[features]
y = df[['Price']]

# Preprocessing
cat_features = [col for col in ['Make', 'Model', 'Trim', 'Type'] if col in features]
num_features = [col for col in features if col not in cat_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ]
)

# Model
model = LinearRegression()
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])

# Train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
pipeline.fit(x_train, y_train)

# Calculate metrics
y_pred = pipeline.predict(x_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

def price_pred(make, model, trim, mileage, car_type, cylinder, liter, doors, cruise, sound, leather):
    input_data = pd.DataFrame({
        'Make': [make],
        'Model': [model],
        'Trim': [trim],
        'Mileage': [mileage],
        'Type': [car_type],
        'Cylinder': [cylinder],
        'Liter': [liter],
        'Doors': [doors],
        'Cruise': [cruise],
        'Sound': [sound],
        'Leather': [leather]
    })
    # Filter columns to only what the model expects
    input_data = input_data[features]
    prediction = pipeline.predict(input_data)[0][0]
    return prediction

st.title('MLOps Car Price Prediction App :red_car:')
st.write('Enter Car Details to Predict the Price')

make = st.selectbox('Make', df['Make'].unique())
model_options = df[df['Make'] == make]['Model'].unique()
carmodel = st.selectbox('Model', model_options)
trim_options = df[(df['Make'] == make) & (df['Model'] == carmodel)]['Trim'].unique()
# Handle case if rim options is empty (though unlikely with this dataset structure)
if len(trim_options) > 0:
    trim = st.selectbox('Trim', trim_options)
else:
    trim = st.selectbox('Trim', df['Trim'].unique()) # Fallback

mileage = st.number_input('Mileage', min_value=200, max_value=60000, step=100)
car_type = st.selectbox('Type', df['Type'].unique())
cylinder = st.selectbox('Cylinder', df['Cylinder'].unique())
liter = st.number_input('Liter', min_value=0.0, max_value=12.0, step=0.1, value=1.6)
doors = st.selectbox('Doors', df['Doors'].unique())

cruise = st.radio('Cruise', [0, 1])
sound = st.radio('Sound', [0, 1])
leather = st.radio('Leather', [0, 1])

if st.button('Predict'):
    predicted_price = price_pred(make, carmodel, trim, mileage, car_type, cylinder, liter, doors, cruise, sound, leather)
    price = float(predicted_price)
    st.success(f'The Predicted price of the car is: ${price:,.2f}')
    st.success(f'The R2 Score is : {r2:.2f}')
    st.success(f'The RMSE Score is : {rmse:.2f}')
