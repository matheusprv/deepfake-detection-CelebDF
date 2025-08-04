import tensorflow as tf
import os
import cv2
import numpy as np
import shutil

from tensorflow.keras.utils import plot_model

def read_video(video_path):
    video_frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, current_frame = cap.read()
        if not ret: break
        current_frame = cv2.resize(current_frame, (299, 299))
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        video_frames.append(current_frame)
    cap.release()

    video_frames = np.array([video_frames]) / 255.

    return video_frames



model = tf.keras.models.load_model('celeb_df_best_model_SGD.keras')
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True, show_layer_activations=True)


TEST_PATH = "/media/work/matheusvieira/deep_fake_detection/Celeb-df-V2-faces-extracted/test"

real_videos = os.listdir(TEST_PATH + "/real")
fake_videos = os.listdir(TEST_PATH + "/fake")

real_videos_path = [(os.path.join(TEST_PATH + "/real", video_path), np.array([1., 0.])) for video_path in real_videos]
fake_videos_path = [(os.path.join(TEST_PATH + "/fake", video_path), np.array([0., 1.])) for video_path in fake_videos]

dataset = real_videos_path + fake_videos_path



wrong_videos_path = []

for i, (video_path, label) in enumerate(dataset):
    video = read_video(video_path)
    prediction = model.predict(video, verbose = 0)

    if np.argmax(prediction) != np.argmax(label):
        wrong_videos_path.append(video_path)

    if (i+1) % 50 == 0:
        print(f"{i+1}", end = " | ")

total_videos = len(dataset)
wrong = len(wrong_videos_path)
right = total_videos - wrong

print("Total videos:", total_videos)
print("Correct Videos:", right)
print("Wrong videos:", wrong)



dirpath = "/media/work/matheusvieira/deep_fake_detection/VideoTrain/wrong_classification"
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)

os.makedirs(dirpath + "/real")
os.makedirs(dirpath + "/fake")

for path in wrong_videos_path:
    if "real" in path:
        shutil.copy(src = path, dst = (dirpath + "/real"))
    else:
        shutil.copy(src = path, dst = (dirpath + "/fake"))