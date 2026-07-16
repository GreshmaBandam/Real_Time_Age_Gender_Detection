import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from src.dataset_loader import UTKFaceDataset
from src.preprocess import train_transform, split_dataset
from src.model import AgeGenderCNN

# -----------------------------
# Device Configuration
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)


# -----------------------------
# Load Dataset
# -----------------------------
dataset = UTKFaceDataset(
    image_dir="dataset/UTKFace",
    transform=train_transform
)

train_set, val_set, test_set = split_dataset(dataset)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)


# -----------------------------
# Model
# -----------------------------
model = AgeGenderCNN().to(device)

age_loss_fn = nn.CrossEntropyLoss()
gender_loss_fn = nn.CrossEntropyLoss()

optimizer = Adam(model.parameters(), lr=0.001)

epochs = 10

os.makedirs("saved_models", exist_ok=True)

best_loss = float("inf")


# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(epochs):

    model.train()

    total_loss = 0

    progress = tqdm(train_loader)

    for images, ages, genders in progress:

        images = images.to(device)
        ages = ages.to(device)
        genders = genders.to(device)

        optimizer.zero_grad()

        age_pred, gender_pred = model(images)

        age_loss = age_loss_fn(age_pred, ages)
        gender_loss = gender_loss_fn(gender_pred, genders)

        loss = age_loss + gender_loss

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        progress.set_description(
            f"Epoch {epoch+1}/{epochs}"
        )

        progress.set_postfix(
            loss=loss.item()
        )

    avg_loss = total_loss / len(train_loader)

    print(f"\nEpoch {epoch+1} Loss : {avg_loss:.4f}")

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            "saved_models/age_gender_model.pth"
        )

        print("Model Saved Successfully!")

print("\nTraining Completed!")