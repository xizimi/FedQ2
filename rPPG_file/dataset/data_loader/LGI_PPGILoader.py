"""The dataloader for LGI-PPGI datasets.

"""
import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
from rPPG_file.dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
import xml.etree.ElementTree as ET

def sample(a, len):
    """Samples a sequence into specific length."""
    return np.interp(
        np.linspace(
            1, a.shape[0], len), np.linspace(
            1, a.shape[0], a.shape[0]), a)

class LGI_PPGILoader(BaseLoader):
    """The data loader for the LGI-PPG dataset."""

    def __init__(self, name, data_path, config_data,i):
        """Initializes an LGI_PPGI dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- id1_1/
                     |       |-- cv_camera_sensor_stream_handler.avi
                     |       |-- cv_camera_sensor_timer_stream_handler.xml
                     |   |-- id1_2/
                     |       |-- cv_camera_sensor_stream_handler.avi
                     |       |-- cv_camera_sensor_timer_stream_handler.xml
                     |   |-- id2_1/
                     |       |-- cv_camera_sensor_stream_handler.avi
                     |       |-- cv_camera_sensor_timer_stream_handler.xml
                     |...
                     |   |-- id6_2/
                     |       |-- cv_camera_sensor_stream_handler.avi
                     |       |-- cv_camera_sensor_timer_stream_handler.xml
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data,i)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For LGI-PPG dataset)."""
        data_dir1 = glob.glob(data_path + os.sep + "id*")
        data_dir2=[]
        for m in data_dir1:
            for n in os.listdir(m):
                data_dir2.append(os.path.join(m,n))
        data_dirs=[]
        for j in data_dir2:
            for k in os.listdir(j):
                data_dirs.append(os.path.join(j,k))
        # print(data_dirs)
        if not data_dirs:
            raise ValueError(self.name + " dataset get data error!")
        dirs = list()
        for data_dir in data_dirs:
            subject_trail_val = os.path.split(data_dir)[-1]
            index = subject_trail_val
            dirs.append({"index": index, "path": data_dir})
        return dirs

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i,file_list_dict):
        """   invoked by preprocess_dataset for multi_process.   """
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']
        frames = self.read_video(
            os.path.join(
                data_dirs[i]['path'],
                "cv_camera_sensor_stream_handler.avi"))
        bvps = self.read_wave(
            os.path.join(
                data_dirs[i]['path'],
                "cms50_stream_handler.xml"))
        bvps = sample(bvps, frames.shape[0])
        frames_clips, bvps_clips = self.preprocess(
            frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list
    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T,H,W,3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        tree = ET.parse(bvp_file)
        # get all bvp elements and their values
        bvp_elements = tree.findall('.//*/value2')
        bvp = [int(item.text) for item in bvp_elements]
        return np.asarray(bvp)
