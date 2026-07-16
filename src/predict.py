import torch
from PIL import Image
from torchvision import transforms

from src.model import AgeGenderCNN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Labels
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

# Transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# Load Model
model = AgeGenderCNN()
model.load_state_dict(torch.load(
    "saved_models/age_gender_model.pth",
    map_location=device
))

model.to(device)
model.eval()


def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        age_output, gender_output = model(tensor)

        age = torch.argmax(age_output, dim=1).item()

        gender = torch.argmax(gender_output, dim=1).item()

    return AGE_GROUPS[age], GENDERS[gender]


if __name__ == "__main__":

    age, gender = predict("test.jpg")

    print("Age :", age)
    print("Gender :", gender)