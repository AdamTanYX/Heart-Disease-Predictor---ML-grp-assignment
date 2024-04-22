import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open('model_ab', 'rb') as f:
    model = pickle.load(f)

# Set page title and layout
st.set_page_config(page_title="Heart Disease Prediction App", layout="wide")

# Add a custom CSS stylesheet
st.markdown("""
<style>
.sidebar .sidebar-content {
    background-color: #f8f9fa;
}
</style>
""", unsafe_allow_html=True)

# Set app title and description
st.write("""
# Simple Heart Disease Prediction App
This app predicts the chances of getting Heart Diseases!
""")

# Add sidebar header
st.sidebar.header('User Input Parameters')

# Function to get user input features
def user_input_features():
    Age = st.sidebar.number_input('Age')
    Sex = st.sidebar.selectbox('Sex', options=['F', 'M'])
    ChestPainType = st.sidebar.selectbox("Chest Pain Type", ['TA', 'ATA', 'NAP', 'ASY'])
    RestingBP = st.sidebar.number_input('Resting Blood Pressure')
    Cholesterol = st.sidebar.number_input('Cholesterol')
    FastingBS = st.sidebar.selectbox("Fasting Blood Sugar", [0, 1])
    RestingECG = st.sidebar.selectbox("Resting Electrocardiogram Results", ['Normal', 'ST', 'LVH'])
    MaxHR = st.sidebar.slider("Maximum Heart Rate Achieved\n(Numeric value between 60 and 202)", min_value=60, max_value=202)
    ExerciseAngina = st.sidebar.selectbox('Exercise-induced Angina', options=['Y', 'N'])
    Oldpeak = st.sidebar.number_input('Oldpeak')
    ST_Slope = st.sidebar.selectbox('The Slope of the Peak Exercise ST Segment', options=['Up', 'Flat', 'Down'])

    data = {
        'Age': Age,
        'Sex': Sex,
        'ChestPainType': ChestPainType,
        'RestingBP': RestingBP,
        'Cholesterol': Cholesterol,
        'FastingBS': FastingBS,
        'RestingECG': RestingECG,
        'MaxHR': MaxHR,
        'ExerciseAngina': ExerciseAngina,
        'Oldpeak': Oldpeak,
        'ST_Slope': ST_Slope
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Get user input features
df = user_input_features()

df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
df['ChestPainType'] = df['ChestPainType'].map({'TA': 1, 'ATA': 2, 'NAP': 3, 'ASY': 4})
df['RestingECG'] = df['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2})
df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
df['ST_Slope'] = df['ST_Slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})

# Create an instance of StandardScaler
scaler = StandardScaler()

df_scaled = scaler.fit_transform(df)

# Make predictions using the loaded model
predictions = model.predict(df)

# Display prediction result
st.write("""
## Prediction
""")

if st.button('Predict'):
    st.markdown(predictions)
    if predictions == 0:
        st.markdown("<h2 style='color: green;'>There is a low chance of getting heart diseases.</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color: red;'>There is a high chance of getting heart diseasses.</h2>", unsafe_allow_html=True)
