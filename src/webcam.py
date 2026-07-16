import cv2
import torch
from PIL import Image
from torchvision import transforms

from src.model import AgeGenderCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AGE_GROUPS = [
    "0-2","3-9","10-19","20-29",
    "30-39","40-49","50-69","70+"
]

GENDERS = ["Male","Female"]

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

model = AgeGenderCNN()
model.load_state_dict(
    torch.load(
        "saved_models/age_gender_model.pth",
        map_location=device
    )
)
model.to(device)
model.eval()

face_detector = cv2.CascadeClassifier(
    "models/haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )

    for (x,y,w,h) in faces:

        face = frame[y:y+h, x:x+w]

        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(rgb)

        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():

            age_pred, gender_pred = model(tensor)

            age = AGE_GROUPS[
                torch.argmax(age_pred).item()
            ]

            gender = GENDERS[
                torch.argmax(gender_pred).item()
            ]

        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"{gender}, {age}",
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

    cv2.imshow(
        "Real-Time Age & Gender Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()