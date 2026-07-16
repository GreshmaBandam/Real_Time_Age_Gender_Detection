import torch
import torch.nn as nn


class AgeGenderCNN(nn.Module):

    def __init__(self):
        super(AgeGenderCNN, self).__init__()

        # Feature Extractor
        self.features = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.flatten = nn.Flatten()

        self.shared = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Age Prediction Head
        self.age_head = nn.Linear(512, 8)

        # Gender Prediction Head
        self.gender_head = nn.Linear(512, 2)

    def forward(self, x):

        x = self.features(x)
        x = self.flatten(x)
        x = self.shared(x)

        age = self.age_head(x)
        gender = self.gender_head(x)

        return age, gender