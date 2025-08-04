import tensorflow as tf
import cv2
import numpy as np
import os
import random


def manipulate_video(video: list):

    video = np.array(video)

    brightness_factor = random.randint(-20, 20)
    contrast_factor = random.uniform(0.5, 1.5)
    saturation_factor = random.uniform(0.5, 1.3)
    hue_shift = random.randint(-10, 10)
    noise_factor = lambda: random.uniform(.25, .35)

    manipulated_video = np.empty_like(video)

    for i in range(video.shape[0]):
        frame = video[i]

        # Increase brightness and contrast
        frame = cv2.convertScaleAbs(frame, alpha = contrast_factor, beta = brightness_factor)

        # Convert to HSV to adjust saturation and hue
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Increase saturation
        hsv_frame[..., 1] = hsv_frame[..., 1] * saturation_factor

        # Adjust hue
        hsv_frame[..., 0] = (hsv_frame[..., 0] + hue_shift) % 180

        # Clip values to valid range
        hsv_frame = np.clip(hsv_frame, 0, 255)

        # Convert back to RGB
        frame = cv2.cvtColor(hsv_frame.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Add noise
#        noise = np.random.normal(0, noise_factor(), frame.shape).astype(np.uint8)
#        frame = cv2.add(frame, noise)

        # Clip values to valid range again after adding noise
        frame = np.clip(frame, 0, 255)

        manipulated_video[i] = frame

    return manipulated_video

class CustomDataGen(tf.keras.utils.Sequence):
    def __init__(self, data_folder, batch_size, input_size = (24, 299, 299, 3), will_manipulate_video = False):
        self.batch_size = batch_size
        self.input_size = input_size
        self.will_manipulate_video = will_manipulate_video

        self.dataset = self.merge_real_fake(data_folder)
        self.n = len(self.dataset)

        self.current_epoch = 0

    def merge_real_fake(self, data_folder):
        real_folder = os.path.join(data_folder, 'real')
        fake_folder = os.path.join(data_folder, 'fake')

        real_videos = os.listdir(real_folder)
        fake_videos = os.listdir(fake_folder)

        real_videos_path = [(os.path.join(real_folder, video_path), [1., 0.]) for video_path in real_videos]
        fake_videos_path = [(os.path.join(fake_folder, video_path), [0, 1.]) for video_path in fake_videos]

        dataset = real_videos_path + fake_videos_path
        for _ in range(3):
            random.shuffle(dataset)

        return dataset

    def read_video(self, video_path):
        video_frames = []
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, current_frame = cap.read()
            if not ret: break
            current_frame = cv2.resize(current_frame, (299, 299))
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            video_frames.append(current_frame)
        cap.release()

        if self.will_manipulate_video and self.current_epoch % 2 != 0:
            video_frames = manipulate_video(video_frames)

        return video_frames

    def get_input(self, video_path):
        video = self.read_video(video_path)
        return video

    def __get_data(self, batch):
        X = np.asarray([self.get_input(x) for x,_ in batch]) / 255.0
        Y = np.asarray([y for _, y in batch])
        return X, Y

    def __getitem__(self, index):
        begin = self.batch_size * index
        end   = self.batch_size * (index + 1)
        batch = self.dataset[begin : end]
        return self.__get_data(batch)

    def on_epoch_end(self):
        self.current_epoch += 1
        for _ in range(3):
            random.shuffle(self.dataset)

    def __len__(self):
        return self.n // self.batch_size
 
