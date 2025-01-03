import streamlit as st
import pickle as pickle
import numpy as np
from train import (sepal_length_max, sepal_length_min, sepal_length_avg,
                    sepal_width_max, sepal_width_min, sepal_width_avg,
                petal_length_max, petal_length_min, petal_length_avg,
                  petal_width_max, petal_width_min, petal_width_avg)

st.title("Iris_app")

with st.sidebar:
    # Custom title style
    st.markdown("""
        <style>
        .sidebar-title {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
            margin-bottom: 20px;
        }
        .custom-slider {
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-title">Iris Feature Input</div>', unsafe_allow_html=True)

    # Add tooltips and descriptive text
    sepal_length_value = st.slider(
        "Sepal Length (in cm)", 
        sepal_length_min, 
        sepal_length_max, 
        sepal_length_avg, 
        0.1, 
        help="Adjust the slider to set the Sepal Length value (in cm)."
    )
    sepal_width_value = st.slider(
        "Sepal Width (in cm)", 
        sepal_width_min, 
        sepal_width_max, 
        sepal_width_avg, 
        0.1, 
        help="Adjust the slider to set the Sepal Width value (in cm)."
    )
    petal_length_value = st.slider(
        "Petal Length (in cm)", 
        petal_length_min, 
        petal_length_max, 
        petal_length_avg, 
        0.1, 
        help="Adjust the slider to set the Petal Length value (in cm)."
    )
    petal_width_value = st.slider(
        "Petal Width (in cm)", 
        petal_width_min, 
        petal_width_max, 
        petal_width_avg, 
        0.1, 
        help="Adjust the slider to set the Petal Width value (in cm)."
    )
st.write("### Selected Values")
st.write(f"Sepal Length: {round(sepal_length_value, 1)} cm")
st.write(f"Sepal Width: {round(sepal_width_value, 1)} cm")
st.write(f"Petal Length: {round(petal_length_value, 1)} cm")
st.write(f"Petal Width: {round(petal_length_value, 1)} cm")

feature_array = np.array([sepal_length_value, sepal_width_value, petal_length_value, petal_width_value])
rounded_feature_array = np.round(feature_array, 1)
reshaped_feature_array = rounded_feature_array.reshape(1, -1)

# Display the rounded array (optional)
st.write("### Rounded Feature Array")
st.write(reshaped_feature_array)

model = pickle.load(open("model.pkl", "rb"))
prediction = model.predict(reshaped_feature_array)
st.write(prediction[0])




