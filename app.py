import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import joblib

gujarati_consonants_dict = {
    'k': 'ક', 'kh': 'ખ', 'g': 'ગ', 'gh': 'ઘ', 'ng': 'ઙ',
    'ch': 'ચ', 'chh': 'છ', 'j': 'જ', 'z': 'ઝ',
    'at': 'ટ', 'ath': 'ઠ', 'ad': 'ડ', 'adh': 'ઢ', 'an': 'ણ',
    't': 'ત', 'th': 'થ', 'd': 'દ', 'dh': 'ધ', 'n': 'ન',
    'p': 'પ', 'f': 'ફ', 'b': 'બ', 'bh': 'ભ', 'm': 'મ',
    'y': 'ય', 'r': 'ર', 'l': 'લ', 'v': 'વ', 'sh': 'શ',
    'shh': 'ષ', 's': 'સ', 'h': 'હ', 'al': 'ળ', 'ks': 'ક્ષ',
    'gn': 'જ્ઞ'
}
gujarati_vowels_dict = {'a': 'આ', 'i': 'ઇ', 'ii': 'ઈ', 'u': 'ઉ',
    'oo': 'ઊ', 'ri': 'ઋ', 'rii': 'ૠ', 'e': 'એ', 'ai': 'ઐ',
    'o': 'ઓ', 'au': 'ઔ', 'amn': 'અં', 'ah': 'અઃ',"ru" : "અૃ","ra" : "અ્ર",
    'ar' : "્રઅ"
}

def main():
    # Placeholder for heading
    st.title("Gujarati Handwritten Character Recognizer By Snehal Shukla!!!")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    # Select model
    selected_model = st.selectbox("Select Model", ["Character Model", "Consonant Model", "Vowel Model"])

    # Load selected model
    model, label_dencoder = load_selected_model(selected_model)

    # Make predictions
    if st.button("Predict"):
        predict_image(uploaded_file, model, label_dencoder)

def load_selected_model(selected_model):
    model_path = ""
    label_encoder_path = ""
    
    if selected_model == "Character Model":
        model_path = "char_model_v1.h5"
        label_encoder_path = "char_label_encoder_v1.joblib"
    elif selected_model == "Consonant Model":
        model_path = "con_model_v1.h5"
        label_encoder_path = "con_label_encoder_v1.joblib"
    elif selected_model == "Vowel Model":
        model_path = "vow_model_v1.h5"
        label_encoder_path = "vow_label_encoder_v1.joblib"

    model = load_model(model_path)
    label_dencoder = joblib.load(label_encoder_path)

    return model, label_dencoder

def predict_image(uploaded_file, model, label_dencoder):
    if uploaded_file is not None:
        # Preprocess the image
        image = Image.open(uploaded_file)
        image = image.resize((50, 50))
        image = image.convert('L')
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Make prediction
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction)

        # Decode the predicted class using label encoder
        class_label = label_dencoder.inverse_transform([predicted_class])[0]
        guj_class_label = get_gujarati_label(class_label)

        # Display results
        st.subheader("Prediction Results")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if guj_class_label=="":
            st.write("Predicted Class:", f"{class_label}")
        else:
            st.write("Predicted Class:", f"{class_label}")
            st.write("In Gujarati :",f"{guj_class_label}")
        st.write("Confidence:", f"{prediction[0][predicted_class] * 100:.2f}%")
    else:
        st.warning("Please upload an image before predicting.")

def get_gujarati_label(class_label):
    guj_class_label = ""
    if class_label.lower() in gujarati_consonants_dict.keys():
        guj_class_label = gujarati_consonants_dict[class_label.lower()]
    elif class_label.lower() in gujarati_vowels_dict.keys():
        guj_class_label = gujarati_vowels_dict[class_label.lower()]

    return guj_class_label

if __name__ == "__main__":
    main()

