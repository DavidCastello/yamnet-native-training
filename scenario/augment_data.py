import os
import glob
import random
import shutil
import csv
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

def add_noise(wav_data, noise_factor):

    # Generate noise signal with the same shape as input waveform
    noise = np.random.normal(0, 1, len(wav_data))

    # Scale noise signal with the permissible noise factor value
    noise *= noise_factor

    # Add noise signal to input waveform
    augmented_wav_data = wav_data + noise

    # Normalize the augmented waveform data
    augmented_wav_data = librosa.util.normalize(augmented_wav_data)

    return augmented_wav_data

def time_shift(audio, p):
    """
    Shift audio to the left or right by a random amount.
    """
    # Calculate the length of the audio array
    length = audio.shape[0]

    # Calculate the maximum number of samples to shift
    max_shift = int(length * p)

    # Generate a random shift value
    shift = random.randint(-max_shift, max_shift)

    # Create an empty array with the same shape as the audio array
    shifted_audio = np.zeros_like(audio)

    # Shift the audio by the specified number of samples
    if shift > 0:
      # Shift to the right
        shifted_audio[shift:] = audio[:length-shift]
    else:
        # Shift to the left
        shifted_audio[:length+shift] = audio[-shift:]
    
    if np.sum(shifted_audio) == 0:
        #revert the process if all information was erased
        shifted_audio = audio     

    return shifted_audio

def time_stretching(audio,factor,sr):
    
    wav_time_stch = librosa.effects.time_stretch(audio,rate=factor)
    
    return wav_time_stch[:sr*5]

def augment_dataset(args):
    # Get all .wav files in the train directory
    files = glob.glob(os.path.join(args.train_dir, '*', '*.wav'))
    total_files = len(files)
    pbar = tqdm(total=total_files, ncols=100)

    # Process each file
    for file in files:
        # Get file name, base name (without extension), and directory name
        file_name = os.path.basename(os.path.normpath(file))
        file_base_name = os.path.splitext(file_name)[0]
        dir_name = os.path.basename(os.path.dirname(file))

        # Load audio file with librosa, at sampling rate 44100
        data, sr = librosa.load(file, sr=44100)

        # Perform augmentation
        temp_wav = add_noise(data, 0.015)
        temp_wav = time_shift(temp_wav, 0.3)
        temp_wav = time_stretching(temp_wav, 0.85, sr)

        # Create new file name for augmented file
        new_file_name = file_base_name + '_augmented' + '.wav'

        # Define the output path for the augmented file
        output_dir = os.path.join(args.train_dir, dir_name)
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, new_file_name)

        # Write the augmented data to the output file
        sf.write(output_file_path, temp_wav, sr, format='WAV', subtype='PCM_16')

        pbar.update(1)

    pbar.close()