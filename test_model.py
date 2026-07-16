import torch
from src.model import AgeGenderCNN

model = AgeGenderCNN()

dummy = torch.randn(1, 3, 128, 128)

age, gender = model(dummy)

print("Age Output Shape :", age.shape)
print("Gender Output Shape :", gender.shape)