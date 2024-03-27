import streamlit as st
from PIL import Image
import numpy as np
from keras.preprocessing import image
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('loaded_model.h5')

# Define class labels
class_labels = ["Black-grass", "Charlock", "Cleavers", "Common Chickweed", "Common wheat", "Fat Hen",
                "Loose Silky-bent", "Maize", "Scentless Mayweed", "Shepherd's Purse",
                "Small-flowered Cranesbill", "Sugar beet"]

def predict_class(img):
    # Load and preprocess the image
    img = img.resize((224, 224))  # Resize image to match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match batch size
    img_array = img_array / 255.0  # Normalize the image

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]
    confidence = prediction[0][predicted_class_index]

    return predicted_class, confidence

def main():
    st.title("Plant Species Classification")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Predict the class
        predicted_class, confidence = predict_class(image)

        # Display the prediction result
        st.write("Predicted class:", predicted_class)
        st.write("Confidence:", confidence)

        # Display additional information based on the predicted class
        if predicted_class == "Black-grass":
            st.write("Primarily considered a troublesome weed in agriculture, with some traditional herbal remedies using it for its purported diuretic and anti-inflammatory properties.")
        elif predicted_class == "Charlock":
            st.write("Seeds have been used in traditional medicine for expectorant, diuretic, and emetic properties, as well as anti-inflammatory and analgesic effects.")
        elif predicted_class == "Cleavers":
            st.write("Traditionally used as a diuretic and lymphatic tonic, also used topically for skin conditions like eczema and psoriasis.")
        elif predicted_class == "Common Chickweed":
            st.write("Rich in vitamins and minerals, used traditionally as a demulcent and emollient, often used topically to soothe skin irritations and inflammations.")
        elif predicted_class == "Common wheat":
            st.write("Not typically used for medicinal purposes, but some wheat-derived compounds may have health benefits such as antioxidant properties.")
        elif predicted_class == "Fat Hen":
            st.write("Has been used traditionally for its purported diuretic, laxative, and anti-inflammatory properties.")
        elif predicted_class == "Loose Silky-bent":
            st.write("Primarily considered a pasture grass and is not commonly used for medicinal purposes.")
        elif predicted_class == "Maize":
            st.write("Not typically used for medicinal purposes, but various parts of the plant, such as corn silk, have been used in traditional medicine for diuretic and anti-inflammatory properties.")
        elif predicted_class == "Scentless Mayweed":
            st.write("Used in herbal medicine as a mild sedative, digestive aid, and topically for its anti-inflammatory and antiseptic properties.")
        elif predicted_class == "Shepherd's Purse":
            st.write("Used traditionally for its astringent properties and to support women's health, particularly for menstrual irregularities and bleeding disorders.")
        elif predicted_class == "Small-flowered Cranesbill":
            st.write("Used in herbal medicine for its astringent and styptic properties, traditionally used to stop bleeding and promote wound healing.")
        elif predicted_class == "Sugar beet":
            st.write("Primarily cultivated for sugar production, with some research suggesting potential health benefits of sugar beet fiber for promoting digestive health and supporting weight management.")
        else:
            st.write("Information not available for this plant.")

if __name__ == '__main__':
    main()
