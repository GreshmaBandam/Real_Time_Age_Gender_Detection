import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def age_to_group(age):
    """Convert age into one of 8 age groups."""
    if age <= 2:
        return 0
    elif age <= 9:
        return 1
    elif age <= 19:
        return 2
    elif age <= 29:
        return 3
    elif age <= 39:
        return 4
    elif age <= 49:
        return 5
    elif age <= 69:
        return 6
    else:
        return 7


class UTKFaceDataset(Dataset):

    def __init__(self, image_dir, transform=None):

        self.image_dir = image_dir
        self.transform = transform
        self.image_files = []

        for file in os.listdir(image_dir):

            if file.endswith(".jpg"):

                try:
                    age = int(file.split("_")[0])
                    gender = int(file.split("_")[1])

                    self.image_files.append((file, age, gender))

                except:
                    continue

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):

        file, age, gender = self.image_files[index]

        image_path = os.path.join(self.image_dir, file)

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        age_group = age_to_group(age)

        return image, age_group, gender