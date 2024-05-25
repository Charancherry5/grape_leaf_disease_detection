import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
model = tf.keras.models.load_model('grape_leaf_model.h5')

# Function to make a prediction
def predict_disease(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    class_names = ['Black Rot', 'Esca (Black Measles)', 'Healthy', 'Leaf Blight (Isariopsis Leaf Spot)']

    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    return predicted_class, confidence

# Function to get pesticide recommendations
def get_pesticide_recommendation(disease):
    recommendations = {
        'Black Rot': 'Use fungicides such as myclobutanil or mancozeb.',
        'Esca (Black Measles)': 'Use fungicides like tebuconazole or carbendazim. Prune affected areas.',
        'Healthy': 'No pesticide needed. Keep monitoring the plant.',
        'Leaf Blight (Isariopsis Leaf Spot)': 'Use fungicides like chlorothalonil or copper-based sprays.'
    }
    return recommendations.get(disease, 'No recommendation available.')

# Streamlit app
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 48px;
        color: #4CAF50;
    }
    .subtitle {
        text-align: center;
        font-size: 24px;
        color: #777;
    }
    .result {
        font-size: 32px;
        color: #FF6347;
    }
    .recommendation {
        font-size: 28px;
        color: #008080;
    }
    .footer {
        text-align: center;
        font-size: 16px;
        color: #aaa;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">🍇 Grape Leaf Disease Detection and pestiside recomendation</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="subtitle">Upload an image of a grape leaf to detect its disease.</h3>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img.thumbnail((400, 400))  # Resize the image to be smaller while maintaining the aspect ratio
    st.image(img, caption='Uploaded Image', use_column_width=False, width=300)  # Display the smaller image
    st.markdown("### 🔍 Analyzing the image...")
    st.markdown(":rainbow[CNN model is running]")

    predicted_class, confidence = predict_disease(img)
    st.markdown(f'<p class="result">🧬 Prediction: <strong>{predicted_class}</strong></p>', unsafe_allow_html=True)
    st.markdown(f'<p class="result">🔮 Confidence: <strong>{confidence:.2f}</strong></p>', unsafe_allow_html=True)
    
    recommendation = get_pesticide_recommendation(predicted_class)
    st.markdown(f'<p class="recommendation">🌿 Pesticide Recommendation: <strong>{recommendation}</strong></p>', unsafe_allow_html=True)

    if predicted_class != 'Healthy':
        st.markdown(f'<p class="result">⚠️ Take action to treat the <strong>{predicted_class}</strong> disease!</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="result">🎉 Your grape leaf is healthy! Keep up the good work! 💪</p>', unsafe_allow_html=True)

st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('<p class="footer">Developed by <a href="https://github.com/Charancherry5" target="_blank">cherry</a></p>', unsafe_allow_html=True)
