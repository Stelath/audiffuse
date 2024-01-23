import numpy as np
import multiprocessing
import os
import librosa
from tqdm import tqdm
from functools import partial

output_folder_path = ''

def process_audio_file(input_file_path, sample_rate):
    # Load audio file
    audio_waveform, _ = librosa.load(input_file_path, sr=sample_rate)

    # Get output file path
    output_file_path = os.path.join(output_folder_path, os.path.basename(input_file_path).replace('.mp3', '.npy'))

    # Save audio file as numpy array
    with open(output_file_path, 'wb') as f:
        np.save(f, audio_waveform)

def process_audio_files(input_folder_path, output_folder_path, sample_rate):
    # Get list of input file paths
    input_file_paths = [os.path.join(input_folder_path, f) for f in os.listdir(input_folder_path) if f.endswith('.mp3')]

    # Create output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)
    
    # Create a file to store the sample rate
    with open(os.path.join(output_folder_path, 'sample_rate.txt'), 'w') as f:
        f.write(str(sample_rate))

    # Process audio files in parallel using multiprocessing
    with multiprocessing.Pool() as pool:
        for _ in tqdm(pool.imap_unordered(partial(process_audio_file, sample_rate=sample_rate), input_file_paths), total=len(input_file_paths)):
            pass

def remove_decimal_zero(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if '.0' in file:
                file_path = os.path.join(root, file)
                new_file_path = file_path.replace('.0', '')
                os.rename(file_path, new_file_path)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process audio files using librosa and multiprocessing')
    parser.add_argument('-f', '--input-folder', type=str, required=True, help='Path to input folder containing audio files')
    parser.add_argument('-o', '--output-folder', type=str, required=True, help='Path to output folder for saving numpy files')
    parser.add_argument('-r', '--resample', type=int, default=48000, help='Resample audio to this sample rate')
    parser.add_argument('-rd', '--remove-decimal', action='store_true', help='Remove decimal zero from file name')
    args = parser.parse_args()

    output_folder_path = args.output_folder

    print(f"Processing Audio with {multiprocessing.cpu_count()} Workers")

    if args.remove_decimal:
        remove_decimal_zero(args.input_folder)
    else:
        process_audio_files(args.input_folder, args.output_folder, args.resample)