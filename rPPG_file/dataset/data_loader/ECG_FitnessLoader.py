"""The dataloader for UBFC-rPPG dataset.

Details for the UBFC-rPPG Dataset see https://sites.google.com/view/ybenezeth/ubfcrppg.
If you use this dataset, please cite this paper:
S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters, 2017.
"""
import os
import sys
from multiprocessing import Pool, Process, Value, Array, Manager
import cv2
import numpy as np
# from dataset.data_loader.BaseLoader import BaseLoader
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal
from tqdm import tqdm
from rPPG_file.dataset.data_loader.BaseLoader import BaseLoader


class ECG_FitnessLoader(BaseLoader):
    """The data loader for the UBFC-rPPG dataset."""

    def __init__(self, name, data_path, config_data,i):
        super().__init__(name, data_path, config_data,i)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For UBFC-rPPG dataset)."""
        data_dir1=[]
        s=["00","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16"]
        for i in s:
            data_dir1.append(os.path.join(data_path,i))
        data_dir2 = []
        for m in data_dir1:
            for n in os.listdir(m):
                data_dir2.append(os.path.join(m, n))
        dirs = [{"index":os.path.split(data_dir)[-1], "path": data_dir} for data_dir in data_dir2]
        # print(dirs)
        return dirs

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

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ invoked by preprocess_dataset for multi_process."""
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        # Read Frames

        frames = self.read_video(
                os.path.join(data_dirs[i]['path'],"c920-1.avi"))

        # Read Labels
        bvps = self.read_wave(
                os.path.join(data_dirs[i]['path'],"viatom-raw.csv"))
        # print(frames)
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
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

    def parse_ppg(self, sig, fs):
        import neurokit2
        wdata = neurokit2.ppg_process(sig, fs)
        wlist = wdata[1]['PPG_Peaks']
        if len(wlist) < len(sig) / fs * 0.3:
            sys.exit("ppg signal error")
        m = wdata[0]['PPG_Rate'].mean() / 60
        ppg_temp = np.zeros_like(sig)
        ppg_temp[wlist] = 1
        ppg_temp = np.convolve(ppg_temp, np.hanning(int(fs / 2)), mode="same")
        ppg_filter = signal.butter(6, [m * 0.5, m * 1.5], fs=fs).filtfilt(ppg_temp)
        return ppg_filter
    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        # data=pd.read_csv(bvp_file)
        # sig=np.asarray(data[' PPG']).T
        a=os.path.split(bvp_file)
        csv_file=os.path.join(a[0],"c920.csv")
        all_ts = np.loadtxt(csv_file, delimiter = ",")
        ts = all_ts[:1800, 0]
        ts_ofs = all_ts[:, 1]
        ecg_ts, ecg_raw = np.asarray(pd.read_csv(bvp_file).iloc[int(ts_ofs[0]):int(ts_ofs[-1]),:2].dropna()).T
        ts_dst = np.arange(ts[0],ts[-1], 1000 / 50)
        bvp=parse_ecg(ecg_raw,125)
        ecg_bvp=np.interp(ts_dst,ecg_ts,bvp)
        return ecg_bvp
def parse_ecg(sig,fs):
    from biosppy.signals import ecg
    _, _, rpeaks, _, _, _, hr = ecg.ecg(signal=sig, sampling_rate=fs, show=False)
    ecg_ppg = np.zeros_like(sig)
    ecg_ppg[rpeaks] = 1
    hr_est = np.mean(hr / 60)
    ecg_ppg = np.convolve(ecg_ppg, np.hanning(int(fs / 2)), mode="same")
    sos = signal.butter(6, [hr_est * 0.5, hr_est * 1.5], btype='bandpass',fs=fs,output='sos')
    ecg_filter= signal.sosfiltfilt(sos, ecg_ppg)
    return ecg_filter
