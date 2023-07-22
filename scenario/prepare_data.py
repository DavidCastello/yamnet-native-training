import os
import glob
import random
import shutil
import csv
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import subprocess


def test_dataset_size(recording_list):
    assert len(recording_list) == 2000 

def test_recordings(recording_list):
    for recording in tqdm.tqdm(recording_list):
        signal, rate = librosa.load('audio/' + recording, sr=None, mono=False)

        assert rate == 44100
        assert len(signal.shape) == 1  # mono
        assert len(signal) == 220500  # 5 seconds
        assert np.max(signal) > 0
        assert np.min(signal) < 0
        assert np.abs(np.mean(signal)) < 0.2  # rough DC offset check

def download_data(args):
    if args.dataset_name == "esc-50":
        tf.keras.utils.get_file('esc-50.zip',
                                'https://github.com/karoldvl/ESC-50/archive/master.zip',
                                cache_dir='./',
                                cache_subdir='dataset-esc50',
                                extract=True)

    if args.dataset_name == "nbad-10":
        repo_url = 'https://github.com/DavidCastello/NBAD.git'
        nbad_dir = './raw-dataset-nbad10'
        try:
            subprocess.run(['git', 'clone', repo_url, nbad_dir], check=True)
            print(f'Successfully cloned repository {repo_url} into {nbad_dir}')
        except subprocess.CalledProcessError:
            print(f'Failed to clone repository {repo_url} into {nbad_dir}')


def view_metadata(args):
    pd_data = pd.read_csv(args.metadata_path)
    pd_data.head()

def create_category_dataset(args):
    
    if args.dataset_name == "esc-50":
        with open(args.metadata_path) as file:
            data = csv.reader(file, delimiter=",")
            next(data)
            for idx, row in enumerate(data):
                # ['filename', 'fold', 'target', 'category', 'esc10', 'src_file', 'take']
                filename = row[0]
                category = row[3]
                category_path = os.path.join(args.dataset_path, category)
                os.makedirs(category_path, exist_ok=True)
                input_path = os.path.join(args.data_path, filename)
                output_path = os.path.join(category_path, filename)
                shutil.copy2(input_path, output_path)
    
    if args.dataset_name == "nbad-10":
        # Specify the source folder and the destination folder
        source_folder = './raw-dataset-nbad10/audio'

        # Use shutil.copytree() to copy the folder
        shutil.copytree(source_folder, args.dataset_path)