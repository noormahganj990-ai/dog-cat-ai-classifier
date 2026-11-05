import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification

st.set_page_config(page_title="Dog vs Cat Classifier", page_icon="🐶")
st.title("🐶 Dog vs Cat Image Classifier 🐱")
st.write("Upload a picture and the AI model will predict whether it shows a dog or a cat.")

# Replace this with your actual Hugging Face model repository.
HF_MODEL_REPO = "YOUR_HF_USERNAME/dog-cat-vit-classifier"


@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(HF_MODEL_REPO)
    model = ViTForImageClassification.from_pretrained(HF_MODEL_REPO)
    model.eval()
    return processor, model


try:
    processor, model = load_model()
except Exception as error:
    st.error(
        "Could not load the model. Check HF_MODEL_REPO and make sure your "
        "Hugging Face model repository exists and is public."
    )
    st.exception(error)
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG, or PNG image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Classify Image", type="primary"):
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)[0]

            predicted_id = int(probabilities.argmax().item())
            id2label = model.config.id2label
            predicted_label = id2label.get(
                predicted_id, id2label.get(str(predicted_id), str(predicted_id))
            )
            confidence = float(probabilities[predicted_id].item()) * 100

            st.success(f"Prediction: {predicted_label}")
            st.metric("Confidence", f"{confidence:.2f}%")

            st.subheader("Class probabilities")
            for class_id, class_name in sorted(
                id2label.items(), key=lambda item: int(item[0])
            ):
                score = float(probabilities[int(class_id)].item()) * 100
                st.write(f"{class_name}: {score:.2f}%")
                st.progress(score / 100)

    except Exception as error:
        st.error("The image could not be processed.")
        st.exception(error)

st.caption("Student project: predictions may be incorrect.")
