import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import matplotlib as mpl
import matplotlib.image as img
from tensorflow.keras.models import load_model

# Define the last convolutional layer name for Grad-CAM
last_conv_layer_name = "Top_Conv_Layer"


# Function to preprocess the input image
def get_img_array(img, size=(224, 224)):
    image = np.array(img)
    resized_image = cv2.resize(image, size)
    resized_image = resized_image.reshape(-1, *size, 3)
    return np.array(resized_image)


# Function to generate Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
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


# Function to save and display Grad-CAM heatmap on the image
def save_and_display_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = np.array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)


# Function to decode predictions into readable class names
def decode_predictions(preds):
    classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    prediction = classes[np.argmax(preds)]
    return prediction


# Function to make a prediction and generate Grad-CAM heatmap
def make_prediction(img, model, last_conv_layer_name=last_conv_layer_name, campath="cam.jpeg"):
    img_array = get_img_array(img)
    preds = model.predict(img_array)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    save_and_display_gradcam(img, heatmap, cam_path=campath)
    return [campath, decode_predictions(preds)]


# Function to preprocess the image and check if it is suitable for brain scan analysis
def preprocess_image(img):
    img_array = np.array(img)
    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        st.error("The uploaded image is not in the expected RGB format.")
        return None
    # Resize to the expected dimensions (224x224)
    resized_img = cv2.resize(img_array, (224, 224))
    return Image.fromarray(resized_img)


# Function to display tumor information based on prediction
def tumor_info(prediction):
    st.markdown("<h1 style='color: black;'>🧠 Brain Tumor Information</h1>", unsafe_allow_html=True)

    if prediction == 'Glioma':
        st.markdown("""
        ### *Glioma*
        **Symptoms:**
        - Headaches: Often severe and may worsen with activity or in the early morning.
        - Seizures: Sudden onset of seizures is common.
        - Cognitive or Personality Changes: Memory problems, confusion, and mood swings.
        - Nausea and Vomiting: Often due to increased pressure in the brain.
        - Weakness or Numbness: In the limbs, often on one side of the body.
        - Vision Problems: Blurred or double vision.
        - Speech Difficulties: Trouble speaking or understanding speech.

        **Recovery Strategies:**
        - Surgical Removal: When possible, surgical resection of the tumor is the first line of treatment.
        - Radiation Therapy: Often used after surgery to kill remaining cancer cells.
        - Chemotherapy: Drugs like temozolomide are commonly used.
        - Targeted Therapy: Drugs that target specific aspects of cancer cells.
        - Physical Therapy: To regain strength and mobility.
        - Speech Therapy: If speech is affected.
        - Psychological Support: Counseling and support groups to cope with emotional changes.
        """, unsafe_allow_html=True)

    elif prediction == 'Pituitary':
        st.markdown("""
        ### *Pituitary Tumors*
        **Symptoms:**
        - Vision Problems: Especially peripheral vision loss.
        - Headaches: Often persistent.
        - Hormonal Imbalance: Causing symptoms like unexplained weight gain or loss, fatigue, and changes in menstrual cycles or sexual function.
        - Growth Issues: Unusual growth in hands, feet, or face in adults (acromegaly).
        - Fatigue: Persistent tiredness and lack of energy.
        - Mood Changes: Depression or anxiety.

        **Recovery Strategies:**
        - Surgical Removal: Transsphenoidal surgery is common to remove the tumor through the nasal passages.
        - Radiation Therapy: Used if surgery is not fully successful or the tumor recurs.
        - Medications: To shrink the tumor or manage hormone levels, like dopamine agonists.
        - Hormone Replacement Therapy: If the pituitary gland's function is compromised.
        - Regular Monitoring: Regular MRI scans and blood tests to monitor the condition.
        """, unsafe_allow_html=True)

    elif prediction == 'Meningioma':
        st.markdown("""
        ### *Meningioma*
        **Symptoms:**
        - Headaches: Persistent and sometimes severe.
        - Seizures: Especially if the tumor affects certain areas of the brain.
        - Vision Problems: Blurred vision or loss of vision.
        - Hearing Loss: If the tumor is near auditory nerves.
        - Memory Loss: Cognitive impairment and memory problems.
        - Weakness in Limbs: Particularly if the tumor is pressing on the spinal cord or brain.

        **Recovery Strategies:**
        - Surgical Removal: The primary treatment if the tumor is accessible.
        - Observation: For small, slow-growing tumors without symptoms, regular monitoring might be recommended.
        - Radiation Therapy: To target residual tumor cells post-surgery or for inoperable tumors.
        - Stereotactic Radiosurgery: A non-invasive method using targeted radiation.
        - Physical Therapy: To recover mobility and strength.
        - Occupational Therapy: To regain daily living skills.
        - Regular Follow-Ups: MRI scans and neurological exams to monitor for recurrence.
        """, unsafe_allow_html=True)

    elif prediction == 'No Tumor':
        st.markdown("""
        ### *No Tumor*
        **No significant findings detected in the image.**
        """, unsafe_allow_html=True)


# Streamlit application for brain tumor detection
def prediction():
    st.markdown("<h1 style='color: black;'>🧠 Brain Tumor Detection 🩺</h1>", unsafe_allow_html=True)

    placeholder = st.empty()

    with placeholder.form("upload"):
        st.markdown("<h2 style='color: black;'>🔬 Provide a Brain X-ray image 🔍</h2>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        submit = st.form_submit_button("Predict")

    if submit and uploaded_file:
        placeholder.empty()
        image = Image.open(uploaded_file)

        # Preprocess the uploaded image
        preprocessed_image = preprocess_image(image)
        if preprocessed_image is None:
            return

        # Load the trained model
        model = load_model("./models/brain_tumor_prediction.h5")

        # Make a prediction and generate Grad-CAM heatmap
        campath, prediction = make_prediction(preprocessed_image, model, campath="123.jpeg")

        # Display the prediction results
        with st.container():
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"<h3 style='color: black;'>🔍 Prediction: {prediction}</h3>", unsafe_allow_html=True)
                st.write(f"<p style='color: black;'>You have {prediction} in your brain.</p>", unsafe_allow_html=True)
            with col2:
                st.markdown("<h3 style='color: black;'>🧬 Total Types</h3>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.write("<p style='color: black;'>1. No Tumor</p>", unsafe_allow_html=True)
                    st.write("<p style='color: black;'>3. Meningioma</p>", unsafe_allow_html=True)
                with c2:
                    st.write("<p style='color: black;'>2. Glioma</p>", unsafe_allow_html=True)
                    st.write("<p style='color: black;'>4. Pituitary</p>", unsafe_allow_html=True)

        # Display the uploaded and predicted images
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.image(preprocessed_image, caption="Uploaded Image")
            with col2:
                test_img = img.imread(campath)
                st.image(test_img, caption="Predicted Image")

        # Provide a download button for the predicted image
        with open('./123.jpeg', "rb") as file:
            st.download_button(
                label="Download Predicted Image",
                data=file,
                file_name="prediction.jpeg",
                mime="image/jpeg"
            )

        # Show detailed information about the predicted tumor type
        tumor_info(prediction)


# Main function to run the Streamlit app
def main():
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Go to", ["Prediction", "Information"])

    if options == "Prediction":
        prediction()
    elif options == "Information":
        st.markdown("<h1 style='color: black;'>🧠 Brain Tumor Information</h1>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
