import cv2
import json
import os
from PIL import Image
import torch

from generativeimage2text.inference import get_model_tokenizer_transforms

# single video inference
#video_path = 'aux_data/videos/5-xGskbsBgI.webm'
video_path = 'aux_data/videos/mCqb6dJBDC8.mp4'

model_name = 'GIT_BASE_VATEX'
model, tokenizer, transforms = get_model_tokenizer_transforms(model_name)

frames = []
cap = cv2.VideoCapture(video_path)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame = transforms(frame).unsqueeze(0).cuda()
    frames.append(frame)
cap.release()

with torch.no_grad():
    result = model({'image': frames})
caption = tokenizer.decode(result['predictions'][0].tolist(), skip_special_tokens=True)
print(caption)
