import os
import torch
import cv2 as cv
import torchvision
import numpy as np
from PIL import Image
from pathlib import Path
from model import RSLRmodel
from torchvision import transforms

# Initialize model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = RSLRmodel().to(device)
model.load_state_dict(torch.load(f="./best_rslr_model.pth"))

# Setup camera
cap = cv.VideoCapture(0)
frame_width, frame_height = 640, 480
cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

# Classes
current_dir_path = Path.cwd()
test_data_path = current_dir_path / "data/test"

test_transforms = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

dataset = torchvision.datasets.ImageFolder(
    root=test_data_path, 
    transform=test_transforms, 
    target_transform=None
)

classes = dataset.classes
print(classes)

# True classes images
alphabet = {}
alphabet_path = current_dir_path / "alphabet"
for letter in os.listdir(alphabet_path):
    letter_char = letter[0]
    letter_path = alphabet_path / letter
    alphabet[letter_char] = Image.open(letter_path).resize((frame_width, frame_height))

cache_size = 100
labels_queue = []

# Transforms
frame_transform = transforms.Compose([
    transforms.Resize(64),  
    transforms.Pad(padding=(
        0, 0, 
        max(0, 64 - 64), 
        max(0, 64 - 64)
    ), fill=255),  
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

while True:    
    # Get frame
    ret, frame = cap.read()
    if not ret:
        print("No frame got. Check camera status.")
        continue 

    # Resize frame to prepare model input
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame_image = Image.fromarray(frame)
    features = frame_transform(frame_image).unsqueeze(dim=0)

    # Make prediction
    pred_indexes = model(features).argmax(dim=1)
    pred_label = classes[pred_indexes.item()]

    # Averaging result over last "cache_size" predictions
    labels_queue.append(pred_label)
    if len(labels_queue) > cache_size:
        labels_queue.pop(0)
        pred_label = max(set(labels_queue), key=labels_queue.count) # most frequent label

    pred_sign_image = alphabet[pred_label]

    # Stack frame and current prediction
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    output = np.hstack((pred_sign_image, frame))
    
    cv.imshow("Camera", output)
    if cv.waitKey(1) == ord("s"):
        break

cap.release()
cv.destroyAllWindows()