import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from src.model import AgeGenderCNN

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Real-Time Age & Gender Detection",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Real-Time Age & Gender Detection")
st.write("Detect Age & Gender from uploaded images or webcam.")

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = AgeGenderCNN()
    model.load_state_dict(
        torch.load(
            "saved_models/age_gender_model.pth",
            map_location=device
        )
    )
    model.to(device)
    model.eval()
    return model

model = load_model()

# -----------------------------
# Face Detector
# -----------------------------
face_detector = cv2.CascadeClassifier(
    "models/haarcascade_frontalface_default.xml"
)

# -----------------------------
# Labels
# -----------------------------
AGE_GROUPS = [
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-69",
    "70+"
]

GENDERS = [
    "Male",
    "Female"
]

# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
st.sidebar.title("🧑‍🤝‍🧑 Age & Gender Detection")
st.sidebar.markdown("### Tech Stack")
st.sidebar.markdown(
    "- PyTorch\n"
    "- OpenCV (Haar Cascade)\n"
    "- Streamlit UI\n"
    "- UTKFace Dataset"
)
st.sidebar.markdown(" ")
st.sidebar.markdown("### Age Groups")
st.sidebar.write(", ".join(AGE_GROUPS))


# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# -----------------------------
# Prediction Function
# -----------------------------
def predict_face(face):

    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(rgb)

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        age_out, gender_out = model(tensor)

        age_prob = torch.softmax(age_out, dim=1)
        gender_prob = torch.softmax(gender_out, dim=1)

        age_idx = torch.argmax(age_prob).item()
        gender_idx = torch.argmax(gender_prob).item()

        age_conf = age_prob[0][age_idx].item()*100
        gender_conf = gender_prob[0][gender_idx].item()*100

    return (
        AGE_GROUPS[age_idx],
        GENDERS[gender_idx],
        age_conf,
        gender_conf
    )

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown(" ")
st.sidebar.markdown("### Select Input")
option = st.sidebar.radio(
    "",
    ["📁 Upload Image", "📷 Webcam"]
)

# ====================================================
# IMAGE UPLOAD
# ====================================================

if option == "📁 Upload Image":

    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is not None:

        image = Image.open(uploaded).convert("RGB")

        #st.image(image, use_container_width=True)
        col1, col2, col3 = st.columns([1,2,1])

        with col2:
            st.image(image, width=300)

        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():

            age_out, gender_out = model(tensor)

            age_idx = torch.argmax(age_out, dim=1).item()
            gender_idx = torch.argmax(gender_out, dim=1).item()

        st.success(f"Age Group : {AGE_GROUPS[age_idx]}")
        st.success(f"Gender : {GENDERS[gender_idx]}")

# ====================================================
# WEBCAM
# ====================================================

else:

    picture = st.camera_input("Take Picture")

    if picture:

        image = Image.open(picture)

        frame = np.array(image)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60,60)
        )

        if len(faces)==0:

            st.error("No face detected!")

        else:

            for (x,y,w,h) in faces:

                face = frame[y:y+h,x:x+w]

                age,gender,age_c,gender_c = predict_face(face)

                cv2.rectangle(
                    frame,
                    (x,y),
                    (x+w,y+h),
                    (0,255,0),
                    2
                )

                label = f"{gender} | {age}"

                cv2.putText(
                    frame,
                    label,
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )

            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            st.image(
                frame,
                use_container_width=True
            )

            st.success(f"Faces Detected : {len(faces)}")