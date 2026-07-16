from src.dataset_loader import UTKFaceDataset
from src.preprocess import train_transform, split_dataset

dataset = UTKFaceDataset(
    image_dir="dataset/UTKFace",
    transform=train_transform
)

train_set, val_set, test_set = split_dataset(dataset)

print("Total Images :", len(dataset))
print("Training :", len(train_set))
print("Validation :", len(val_set))
print("Testing :", len(test_set))