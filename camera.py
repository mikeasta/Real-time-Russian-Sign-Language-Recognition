import os
import time
import torch
import cv2 as cv
from PIL import Image
from pathlib import Path
from model import RSLRmodel

# Initialize model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = RSLRmodel().to(device)
model.load_state_dict(torch.load(f="./models/rslr_model_epoch_50.pth"))

# Setup camera
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 500)

# Classes
current_dir_path = Path.cwd()
test_data_path = current_dir_path / "data/test"

classes = os.listdir(test_data_path)
print(classes)

# True classes images
alphabet = {}
alphabet_path = current_dir_path / "alphabet"
for letter in os.listdir(alphabet_path):
    letter_char = letter[0]
    letter_path = alphabet_path / letter
    alphabet[letter_char] = Image.open(letter_path)

while True:
    ret, frame = cap.read()
    resized_frame = cv.resize(
        src=frame, 
        dsize=(64, 64)
    )
    tensor_frame = torch.Tensor(resized_frame).permute(2, 0, 1).unsqueeze(dim=0).to(device)

    start_time = time.perf_counter()
    res = model(tensor_frame).argmax(dim=1)
    pred_label = classes[res]
    
    if not ret:
        print("No frame got. Check camera status.")

    cv.imshow("Camera", frame)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")
    if cv.waitKey(1) == ord("s"):
        break

cap.release()
cv.destroyAllWindows()