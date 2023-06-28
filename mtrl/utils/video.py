# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Implementation based on Denis Yarats' implementation of [SAC](https://github.com/denisyarats/pytorch_sac).
"""Utility to record the environment frames into a video."""
import os
import imageio
import torchvision.transforms as transforms
import numpy as np
# from numpngw import write_apng
to_pil_image = transforms.ToPILImage()

class VideoRecorder(object):
    def __init__(self, dir_name, height=500, width=500, delay=50):
        """Class to record the environment frames into a video.

        Args:
            dir_name ([type]): directory to save the recording.
            height (int, optional): height of the frame. Defaults to 256.
            width (int, optional): width of the frame. Defaults to 256.
            camera_id (int, optional): id of the camera for recording. Defaults to 0.
            fps (int, optional): frames-per-second for the recording. Defaults to 30.
        """
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.delay=delay
        self.frames = []

    def reset(self):
        """Initialize the recorder.

        Args:
            enabled (bool, optional): should enable the recorder or not. Defaults to True.
        """
        self.frames = []

    def record(self, frame):
        """Record the frames.

        Args:
            env ([type]): environment to record the frames.
        """
        # if frame is None:
        #     assert env is not None
        #     frame = env.render(
        #         mode="rgb_array",
        #         height=self.height,
        #         width=self.width,
        #     )
        self.frames.append(frame)

    def save(self, file_name):
        """Save the frames as video to `self.dir_name` in a file named `file_name`, with "*.png" format.

        Args:
            file_name ([type]): name of the file to store the video frames.
        """
            # path = os.path.join(self.dir_name, file_name)
            # write_apng('{}/{}.png'.format(self.dir_name, file_name),self.frames, delay=50)
            # write_apng(path, self.frames, delay=50)
        imgs = [np.array(to_pil_image(img)) for img in self.frames]
        imageio.mimsave(self.dir_name+f'/{file_name}.gif', imgs)

