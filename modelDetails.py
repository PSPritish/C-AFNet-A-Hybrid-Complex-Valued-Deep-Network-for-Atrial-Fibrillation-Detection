from utils.model_summary import model_summary
from models.complex_resnet import complex_resnet18
from data.dataloader import load_config
from torchvision import models
import torch

# Setup model
config = load_config()
model = models.resnet18()
# Modify the final fully connected layer for binary classification
model.fc = torch.nn.Linear(
    model.fc.in_features, 1
)  # 1 output for binary classification
# Generate summary
model_summary(model, input_size=(3, 128, 128))

model2 = complex_resnet18(config)
model_summary(model2, input_size=(3, 128, 128))
