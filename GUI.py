import joblib
import numpy as np
import streamlit as st
from sklearn.ensemble import ExtraTreesRegressor

st.image('JLU.png')


st.header('ML model for predicting heavy metal adsorption capacity of geopolymer')

model = joblib.load('ML model.joblib')
ss = joblib.load('StandardScaler.joblib')

col1, col2 = st.columns(2)

with col1:
    feature1 = st.number_input(u'$\mathrm{n(SiO2/Al2O3)}$', step=0.01, format='%.2f')
    feature2 = st.number_input(u'$\mathrm{n(Na2O/Al2O3)}$', step=0.01, format='%.2f')
    feature3 = st.number_input(u'$\mathrm{n(Na2O/SiO2)}$', step=0.01, format='%.2f')
    feature4 = st.number_input(u'$\mathrm{n(H2O/Na2O)}$', step=0.01, format='%.2f')
    feature5 = st.number_input(u'$\mathrm{Specific\;surface\;area}$', step=0.01, format='%.2f')
    feature6 = st.number_input(u'$\mathrm{Curing\;temperature}$', step=0.01, format='%.2f')
    feature7 = st.number_input(u'$\mathrm{Curing\;time}$', step=0.01, format='%.2f')

with col2:
    feature8 = st.number_input(u'$\mathrm{Dosage}$', step=0.01, format='%.2f')
    feature9 = st.number_input(u'$\mathrm{pH}$', step=0.01, format='%.2f')
    feature10 = st.number_input(u'$\mathrm{Contact\;time}$', step=0.01, format='%.2f')
    feature11 = st.number_input(u'$\mathrm{Initial\;concentration}$', step=0.01, format='%.2f')
    feature12 = st.number_input(u'$\mathrm{Temperature}$', step=0.01, format='%.2f')
    feature13 = st.number_input(u'$\mathrm{Radius\;of\;hydrated\;heavy\;metal\;ion}$', step=0.01, format='%.2f')
    feature14 = st.number_input(u'$\mathrm{Valence}$', step=0.01, format='%.2f')
    
feature_values = [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12, feature13, feature14]

if st.button('Predict', type='primary'):
    input_data = np.array([feature_values])
    input_data_scaled = ss.transform(input_data)
    prediction = model.predict(input_data_scaled)
    st.success(f'Predicted adsorption capacity: {prediction[0]:.2f} mg/g', icon="✅")
