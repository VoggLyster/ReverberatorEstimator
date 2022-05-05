import numpy as np
import tensorflow as tf
import functools
import time
import matplotlib.pyplot as plt

def tf_float32(x):
    
    """Ensure array/tensor is a float32 tf.Tensor."""
    if isinstance(x, tf.Tensor):
        return tf.cast(x, dtype=tf.float32)  # This is a no-op if x is float32.
    else:
        return tf.convert_to_tensor(x, tf.float32)

def tf_complex64(x):
    
    """Ensure array/tensor is a complex64 tf.Tensor."""
    if isinstance(x, tf.Tensor):
        return tf.cast(x, dtype=tf.complex64)  # This is a no-op if x is float32.
    else:
        return tf.convert_to_tensor(x, tf.complex64)

def compute_mag(audio, size, overlap):
    audio = tf_float32(audio)
    mag = tf.abs(tf.signal.stft(signals=audio,
        frame_length=size,
        frame_step=int(size * (1.0 - overlap)),
        fft_length=size,
        pad_end=True))
    return mag

def get_spectral(x, normalize=False):
    x = tf_complex64(x)
    spectrogram = tf.abs(tf.signal.fft(x))
    spectrogram = spectrogram[:,1:x.shape[1]//2]
    if(normalize):
        spectrogram = spectrogram / tf.norm(spectrogram, axis=1, keepdims=True)
    return tf.math.reduce_mean(spectrogram, axis=0)

def mean_difference(target, value, loss_type='L1', weights=None):
    
    difference = target - value
    weights = 1.0 if weights is None else weights
    loss_type = loss_type.upper()
    if loss_type == 'L1':
        return tf.reduce_mean(tf.abs(difference * weights))
    elif loss_type == 'L2':
        return tf.reduce_mean(difference**2 * weights)

def get_envelope(x, frame_size=4096, overlap=0.5):
    envelope = []
    hop = int((1-overlap)*frame_size)
    
    for i in range(0, x.shape[1], hop):
        tensor_slice = x[:, i:i+frame_size]
        envelope.append(tf.math.reduce_max(tensor_slice))
        
    return np.array(envelope)

def get_standard_deviation(signal, win_size=960, step_size=1, segment_size=24000):
    signal = signal[:,0:segment_size]
    h2 = tf.math.square(signal)

    h2 = tf.concat([tf.zeros([h2.shape[0],win_size//2]), h2], axis=1)

    signal_frames = tf.signal.frame(h2, frame_length=win_size, frame_step=step_size, pad_end=True)

    w = tf.signal.hann_window(win_size, periodic=False)
    w_norm = tf.math.divide(w, tf.math.reduce_sum(w))
    signal_frames = tf.math.multiply(signal_frames, w_norm)

    std_dev = tf.math.sqrt(tf.math.reduce_sum(signal_frames, axis=2))
    std_dev = std_dev[:,0:segment_size]

    return std_dev


def get_echo_density(signal, sample_rate, segment_size=3000):
    signal = signal[:,0:segment_size]
    win_size = int(sample_rate*0.02)
    std_dev = get_standard_deviation(signal, win_size=win_size, step_size=1, segment_size=segment_size)
    
    local_density = tf.where(tf.math.greater(tf.math.abs(signal), std_dev), 1.0, 0.0)

    ## Concat zero padding in start of signal to center t in the middle of the window
    local_density = tf.concat([tf.zeros([local_density.shape[0],win_size//2]), local_density], axis=1)

    local_density_frames = tf.signal.frame(local_density, frame_length=win_size, frame_step=1, pad_end=True)
    w = tf.signal.hann_window(win_size, periodic=False)
    w_norm = tf.math.divide(w, tf.math.reduce_sum(w))
    local_density_frames = tf.math.multiply(local_density_frames, w_norm)

    echo_density = 1.0/0.3173 * tf.math.reduce_sum(local_density_frames, axis=2)
    echo_density = echo_density[:,0:segment_size]

    echo_density = tf.math.reduce_mean(echo_density,axis=0)
    
    return echo_density

class ReverberationLoss(tf.keras.layers.Layer):
    def __init__(self,
            sample_rate = 48000,
            fft_sizes=(8192, 4096, 2048),
            overlap=0.75,
            spectral_loss_type='L1',
            spectral_loss_weight=1.0,
            time_loss_type='L1',
            time_loss_weight=0.0,
            envelope_loss_type='L1',
            envelope_loss_weight=0.0,
            early_envelope_type='L1',
            early_envelope_weight=0.0,
            echo_density_type='L1',
            echo_density_weight=0.0,
            use_multiscale=False,
            name='reverberation_loss'):
        super().__init__(name=name)
        self.sample_rate = sample_rate
        self.fft_sizes = fft_sizes
        self.overlap = overlap
        self.spectral_loss_type = spectral_loss_type
        self.spectral_loss_weight = spectral_loss_weight
        self.time_loss_type = time_loss_type
        self.time_loss_weight = time_loss_weight
        self.envelope_loss_type = envelope_loss_type
        self.envelope_loss_weight = envelope_loss_weight
        self.early_envelope_type = early_envelope_type
        self.early_envelope_weight = early_envelope_weight
        self.echo_density_type = echo_density_type
        self.echo_density_weight = echo_density_weight
        self.use_multiscale = use_multiscale

    def call(self, target_audio, output_audio):
        
        loss = 0.0
        if self.use_multiscale:
            spectral_ops = []

            for size in self.fft_sizes:
                spectral_op = functools.partial(compute_mag, size=size, overlap=self.overlap)
                spectral_ops.append(spectral_op)

            for spectral_op in spectral_ops:
                target_mag = spectral_op(target_audio)
                output_mag = spectral_op(output_audio)

                loss += self.spectral_loss_weight * mean_difference(target_mag, output_mag,
                    self.spectral_loss_type)

        if self.spectral_loss_weight > 0.0:
            target_spectral = get_spectral(target_audio, normalize=False)
            output_spectral = get_spectral(output_audio, normalize=False)
            loss += self.spectral_loss_weight * mean_difference(target_spectral, output_spectral,
                self.spectral_loss_type)
            
        if self.time_loss_weight > 0:
            loss += self.time_loss_weight * mean_difference(target_audio, output_audio,
                self.time_loss_type)
            
        if self.envelope_loss_weight > 0:
            target_envelope = get_envelope(target_audio)
            output_envelope = get_envelope(output_audio)
            loss += self.envelope_loss_weight * mean_difference(target_envelope, 
                output_envelope, self.envelope_loss_type)
            
        if self.early_envelope_weight > 0:
            early_length = 4096
            target_envelope = get_envelope(target_audio[:, 0:early_length])
            output_envelope = get_envelope(output_audio[:, 0:early_length])

            loss += self.envelope_loss_weight * mean_difference(target_envelope, 
                output_envelope, self.early_envelope_type)
    
        if self.echo_density_weight > 0:
            target_echo_density = get_echo_density(target_audio,sample_rate=self.sample_rate)
            output_echo_density = get_echo_density(output_audio,sample_rate=self.sample_rate)
            
            loss += self.echo_density_weight * mean_difference(target_echo_density, 
                output_echo_density, self.echo_density_type)

        return loss
        
def reverberationLoss(sample_rate = 48000,
            fft_sizes=(2048, 1024, 512, 256, 128, 64),
            overlap=0.0,
            spectral_loss_type='L1',
            spectral_loss_weight=1.0,
            time_loss_type='L1',
            time_loss_weight=1.0,
            envelope_loss_type='L1',
            envelope_loss_weight=0.0,
            early_envelope_type='L1',
            early_envelope_weight=0.0,
            echo_density_type='L1',
            echo_density_weight=0.0,
            use_multiscale=False,
            name='reverberation_loss'):

    reverberation_loss = ReverberationLoss(sample_rate=sample_rate,
        fft_sizes=fft_sizes,
        overlap=overlap,
        spectral_loss_type=spectral_loss_type,
        spectral_loss_weight=spectral_loss_weight,
        time_loss_type=time_loss_type,
        time_loss_weight=time_loss_weight,
        envelope_loss_type=envelope_loss_type,
        envelope_loss_weight=envelope_loss_weight,
        early_envelope_type=early_envelope_type,
        early_envelope_weight=early_envelope_weight,
        echo_density_type=echo_density_type,
        echo_density_weight=echo_density_weight,
        use_multiscale=use_multiscale,
        name=name)   
    
    def loss(y_true,y_pred):
        
        return reverberation_loss(y_true, y_pred)
    
    return loss