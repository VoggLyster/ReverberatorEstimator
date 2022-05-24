import tensorflow_io as tfio
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa.display
from librosa.util import fix_length
from ReverberatorEstimator import loss
import numpy as np
from pedalboard import load_plugin
from scipy.io import wavfile
import IPython
import os


def load_audio_file(audio_file_name):
    audio_binary = tf.io.read_file(audio_file_name)
    audio, _ = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio)
    audio_len = tf.size(audio)
    audio = tf.reshape(audio,(audio_len))
    return audio

def get_dataset(dry_file, wet_file, batch_size, resample=False, old_sample_rate=48000, new_sample_rate=16000):
    dry_files = []

    for f in range(batch_size):
        audio = load_audio_file(dry_file)
        
        if resample:
            audio = tfio.audio.resample(audio, old_sample_rate, new_sample_rate)
        dry_files.append(audio)
        
    x_train = tf.stack(dry_files)

    wet_files = []

    for f in range(batch_size):
        audio = load_audio_file(wet_file)
        if resample:
            audio = tfio.audio.resample(audio, old_sample_rate, new_sample_rate)
        wet_files.append(audio)
        
    y_train = tf.stack(wet_files)

    return x_train, y_train

def plot_single(audio, sample_rate, sample_length):
    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(7,25))
    ax[0].plot(audio)
    ax[0].set_title("Time")
    ax[0].set_ylim(-1,1)
    S = np.abs(librosa.stft(audio))
    D = librosa.amplitude_to_db(S, ref=np.max)
    img = librosa.display.specshow(D, y_axis='linear', x_axis='time',
                                sr=sample_rate, ax=ax[1])
    ax[1].set_title("STFT")
    spectral_diff = abs(loss.get_spectral(tf.reshape(audio, (1,sample_length))))
    ax[2].plot(spectral_diff)
    ax[2].set_title("Spectrogram")
    envelope_diff = loss.get_envelope(tf.reshape(audio, (1,sample_length)), frame_size=4096)
    ax[3].plot(envelope_diff)
    ax[3].set_title("Envelope")
    ax[3].set_ylim(0,2)
    echo_density_diff = loss.get_echo_density(tf.reshape(audio, (1,sample_length)), sample_rate=sample_rate) 
    ax[4].plot(echo_density_diff)
    ax[4].set_title("Echo density")
    ax[4].set_ylim(0,2)

def plot_output_and_target(output_audio, target_audio, sample_rate):
    fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(15,25))
    ax[0,0].plot(output_audio.numpy()[0])
    ax[0,0].set_title("Output audio")
    ax[0,0].set_ylim(-1,1)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(output_audio.numpy()[0])), ref=np.max)
    img = librosa.display.specshow(D, y_axis='linear', x_axis='time',
                                sr=sample_rate, ax=ax[1,0])
    ax[1,0].set_title("Spectrogram of output audio")
    spec_loss_output = loss.get_spectral(output_audio)
    max_spec_loss_output = np.max(spec_loss_output)
    ax[2,0].plot(spec_loss_output)
    ax[2,0].set_title("Spectral of output audio")
    ax[3,0].plot(loss.get_envelope(output_audio, frame_size=4096))
    ax[3,0].set_title("Envelope of output audio")
    ax[4,0].plot(loss.get_echo_density(output_audio, sample_rate=sample_rate))
    ax[4,0].set_title("Echo density of output audio")
    ax[0,1].plot(target_audio.numpy()[0])
    ax[0,1].set_title("Target audio")
    ax[0,1].set_ylim(-1,1)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(target_audio.numpy()[0])), ref=np.max)
    img = librosa.display.specshow(D, y_axis='linear', x_axis='time',
                                sr=sample_rate, ax=ax[1,1])
    ax[1,1].set_title("Spectrogram of target audio")
    spec_loss_target = loss.get_spectral(target_audio)
    max_spec_loss_target = np.max(spec_loss_target)
    ax[2,1].plot(spec_loss_target)
    ax[2,1].set_title("Spectral of target audio")
    ax[3,1].plot(loss.get_envelope(target_audio,frame_size=4096))
    ax[3,1].set_title("Envelope of target audio")
    ax[4,1].plot(loss.get_echo_density(target_audio, sample_rate=sample_rate))
    ax[4,1].set_title("Echo density of target audio")
    
    ax[2,0].set_ylim(0, max(max_spec_loss_output, max_spec_loss_target))
    ax[2,1].set_ylim(0, max(max_spec_loss_output, max_spec_loss_target))

def plot_differences(output_audio, target_audio, sample_rate, weights=None):
    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(7,25))
    time_diff = abs(target_audio.numpy()[0] - output_audio.numpy()[0])
    ax[0].plot(time_diff)
    ax[0].set_title("Absolute audio difference")
    ax[0].set_ylim(0,1.5)
    Sx = np.abs(librosa.stft(output_audio.numpy()[0]))
    Sy = np.abs(librosa.stft(target_audio.numpy()[0]))
    D = librosa.amplitude_to_db(Sy-Sx, ref=np.max)
    img = librosa.display.specshow(D, y_axis='linear', x_axis='time',
                                sr=sample_rate, ax=ax[1])
    ax[1].set_title("Spectrogram of audio difference")
    spectral_diff = abs(loss.get_spectral(target_audio) - loss.get_spectral(output_audio))
    ax[2].plot(spectral_diff)
    ax[2].set_title("Spectral difference")
    envelope_diff = abs(loss.get_envelope(target_audio, frame_size=4096) - loss.get_envelope(output_audio, frame_size=4096))
    ax[3].plot(envelope_diff)
    ax[3].set_title("Envelope difference")
    ax[3].set_ylim(0,2)
    echo_density_diff = abs(loss.get_echo_density(target_audio, sample_rate=sample_rate) 
                            - loss.get_echo_density(output_audio, sample_rate=sample_rate))
    ax[4].plot(echo_density_diff)
    ax[4].set_title("Echo density difference")
    ax[4].set_ylim(0,2)

    print('-----------------------------------')
    print('Sub losses: ')
    print('Time loss: ', np.mean(abs(time_diff)))
    print('Spectral loss: ', np.mean(abs(spectral_diff)))
    print('Envelope loss: ', np.mean(abs(envelope_diff)))
    print('Echo density loss: ', np.mean(abs(echo_density_diff)))
    if weights != None:
        print('-----------------------------------')
        print('Weighted losses: ')
        print('Time loss: ', np.mean(abs(time_diff)) * weights[0])
        print('Spectral loss: ', np.mean(abs(spectral_diff)) * weights[1])
        print('Envelope loss: ', np.mean(abs(envelope_diff)) * weights[2])
        print('Echo density loss: ', np.mean(abs(echo_density_diff)) * weights[3])

def generate_MUSHRA_ready_audio(vst_path, parameters, sample_rate=44100):
    vst = load_plugin(vst_path)
    params = np.copy(parameters)
    audio_data = []
    audio_names = []

    seconds = 12
    sample_length = seconds * sample_rate
    
    for root, sub, files in os.walk('./MUSHRA_audio/Dry'):
        files = sorted(files)
        for f in files:
            audio = load_audio_file(os.path.join(root, f))
            audio = fix_length(audio, size=sample_length)
            for i in range(len(params)):
                for j, key in enumerate(vst.parameters.keys()):
                    if j == i:
                        setattr(vst, key, params[i]*100.0)
            audio_data.append(vst.process(audio, sample_rate))
            audio_names.append(f)
            
    return audio_data, audio_names

def generate_MUSHRA_anchor_audio(vst_path, parameters, sample_rate):
    vst = load_plugin(vst_path)
    params = np.copy(parameters)
    audio_data = []
    audio_names = []

    seconds = 12
    sample_length = seconds * sample_rate

    reduced_params = params[0:27]
    for root, sub, files in os.walk('./MUSHRA_audio/Dry'):
        files = sorted(files)
        for f in files:
            audio = load_audio_file(os.path.join(root, f))
            audio = fix_length(audio, size=sample_length)
            for i in range(len(reduced_params)):
                for j, key in enumerate(vst.parameters.keys()):
                    if j == i:
                        setattr(vst, key, reduced_params[i]*100.0)
            audio_data.append(vst.process(audio, sample_rate))
            audio_names.append(f)
            
    return audio_data, audio_names


def display_audio_files(data, sample_rate=44100):
    for d in data:
        IPython.display.display(IPython.display.Audio(d, rate=sample_rate))

def write_audio_files(data, names, folder, sample_rate):
    for i, d in enumerate(data):
        wavfile.write(folder+names[i], sample_rate, d)