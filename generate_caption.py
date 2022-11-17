import csv
import cv2
import json
import os
from PIL import Image
import torch

from generativeimage2text.inference import get_model_tokenizer_transforms, test_git_inference_single_image, test_git_inference_single_video


def single_video_inference(model, tokenizer, transforms, video_path):
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
    return caption


def multiple_videos_inference(model, tokenizer, transforms, videos_path, output_name=None):
    if output_name:
        result = []
        csv_file = open('%s.csv' % output_name, 'w', newline='')
        csv_writer = csv.writer(csv_file)
    
    failed = set()

    for root, _, files in os.walk(videos_path):
        for i, file in enumerate(files):
            filename, _ = os.path.splitext(file)
            video_path = os.path.join(root, file)

            try:
                caption = single_video_inference(model, tokenizer, transforms, video_path)
                print('\n VIDEO %d (%s): %s \n' % (i + 1, filename, caption))
            except:
                failed.add(filename)
                print('\n VIDEO %d (%s): %s \n' % (i + 1, filename, 'FAILED'))
                continue
            
            if output_name:
                result.append({'image_id': filename, 'caption': caption})
                csv_writer.writerow([filename, caption])

    if output_name:
        json_data = json.dumps(result)
        with open('%s.json' % output_name, 'w') as cider_file:
            cider_file.write(json_data)

        csv_file.close()

    print('failed videos:')
    print(failed)


# caption = test_git_inference_single_image('aux_data/images/1.jpg', model_name, '')
# caption = test_git_inference_single_video('aux_data/videos/5-xGskbsBgI.webm', model_name, '')
# print(caption)

model_name = 'GIT_BASE_VATEX'
model, tokenizer, transforms = get_model_tokenizer_transforms(model_name)

# video_path = 'aux_data/videos/5-xGskbsBgI.webm'
# caption = single_video_inference(model, tokenizer, transforms, video_path)
# print(caption)

videos_path = '../../Downloads/vatex/test_videos'
output_name = 'git_base_captions'
multiple_videos_inference(model, tokenizer, transforms, videos_path, output_name)
