import streamlit as st
import joblib
import pandas as pd

# Load preprocessor dan model dari folder artifacts
scaler = joblib.load("preprocessor.pkl")
model = joblib.load("model.pkl")

def make_prediction(features):
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    input_df = pd.DataFrame([features], columns=columns)
    
    # transform pakai standard scaler pada numeric variable
    cols_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
    
    # Prediksi menggunakan model
    prediction = model.predict(input_df)
    return prediction[0]

def main():
    st.title('Heart Attack Prediction')
    st.write('Input parameter dibawah ini')

    # data numerical
    age = st.number_input('Age', min_value=1, max_value=120, value=50)
    trestbps = st.number_input('trestbps', min_value=0, max_value=300, value=120)
    chol = st.number_input('Cholesterol', min_value=0, max_value=600, value=200)
    thalach = st.number_input('Maximum Heart Rate Achieve', min_value=0, max_value=250, value=150)
    oldpeak = st.number_input('oldpeak', min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    # data categorical
    sex = st.selectbox('Sex (0 = Female, 1 = Male)', [0, 1])
    cp = st.selectbox('Chest Pain Type (cp) [0-3]', [0, 1, 2, 3])
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs) [0 = False, 1 = True]', [0, 1])
    restecg = st.selectbox('Resting ECG results (restecg) [0-2]', [0, 1, 2])
    exang = st.selectbox('Exercise Induced Angina (exang) [0 = No, 1 = Yes]', [0, 1])
    slope = st.selectbox('Slope of the peak exercise ST segment (slope) [0-2]', [0, 1, 2])
    ca = st.selectbox('Number of major vessels (ca) [0-4]', [0, 1, 2, 3, 4])
    thal = st.selectbox('Thalassemia (thal) [0-3]', [0, 1, 2, 3])
    
    # Prediction process
    if st.button('Make Prediction'):
        features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        result = make_prediction(features)
        if result == 1:
            st.error(f'The prediction is: {result} (High Risk of Heart Attack)')
        else:
            st.success(f'The prediction is: {result} (Low Risk of Heart Attack)')

if __name__ == '__main__':

    main()
