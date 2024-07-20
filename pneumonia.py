import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
import matplotlib.cm as cm

# Define some functions for prediction
last_conv_layer_name = "Top_Conv_Layer"


def get_img_array(img, size=(224, 224)):
    img = img.convert('L')  # Ensure image is in grayscale
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, size)
    img = np.expand_dims(img, axis=0)
    img = np.array(img)
    return img


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = np.array(img.convert('L'))  # Ensure image is in grayscale
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet", 256)
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)


def decode_predictions(preds):
    classes = ['Normal', 'Pneumonia']
    prediction = classes[np.argmax(preds)]
    return prediction


def make_prediction(img, model, last_conv_layer_name=last_conv_layer_name, campath="cam.jpg"):
    img_array = get_img_array(img, size=(224, 224))
    preds = model.predict(img_array)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    save_and_display_gradcam(img, heatmap, cam_path=campath)
    return [campath, decode_predictions(preds)]


# Prediction base function
def prediction():
    st.header('Pneumonia Detection')

    placeholder = st.empty()

    with placeholder.form("upload"):
        st.header('Provide a Chest X-ray image.')
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        submit = st.form_submit_button("Predict")

    if submit and uploaded_file:
        placeholder.empty()
        image = Image.open(uploaded_file).convert('L')

        model = load_model("./models/pneumonia_prediction.h5")

        campath, prediction = make_prediction(image, model, campath="123.jpg")

        with st.container():
            col1, col2 = st.columns([2, 1])
            with col1:
                st.header(f'Prediction: {prediction}')
                st.write(f'You have {prediction} in your chest.')
            with col2:
                st.header('Total Types')
                st.write('1. Normal')
                st.write('2. Pneumonia')

        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Uploaded Image")
            with col2:
                test_img = plt.imread(campath)
                st.image(test_img, caption="Predicted Image")

        with open('./123.jpg', "rb") as file:
            st.download_button(
                label="Download Predicted Image",
                data=file,
                file_name="prediction.jpg",
                mime="image/jpeg"
            )


# Run the prediction function
if __name__ == '__main__':
    prediction()
