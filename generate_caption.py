from generativeimage2text.inference import test_git_inference_single_image, test_git_inference_single_video

#MODEL = 'GIT_BASE'
MODEL = 'GIT_LARGE_VATEX'

img_path = 'aux_data/images/1.jpg'
video_path = 'aux_data/videos/tomahawk.mp4'

#caption = test_git_inference_single_image(img_path, MODEL, '')
caption = test_git_inference_single_video(video_path, MODEL, '')

print(caption)