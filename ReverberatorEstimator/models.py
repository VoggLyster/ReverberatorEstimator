import tensorflow as tf 
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from ReverberatorEstimator import layers
import numpy as np

def get_simple_models(sample_length, sample_rate, n_parameters, n_processors, vst_path, epsilon, 
    parameter_map, non_learnable_parameters, parameter_model_weights=None):
    """
    Get full model and parameter model

    :param sample_length: length of the input audio
    :param sample_rate: desired sample rate of neural network
    :param n_parameters: number of parameters to be estimated
    :param n_processors: number of processors to be used
    :param vst_path: path to the VST plugin
    :param epsilon: epsilon for the SPSA
    :param n_delay_lines: number of delay lines
    :param true_spsa: use true SPSA or not
    :param linked_EQs: use linked EQs or not
    :param parameter_offset: offset for the parameters (to ignore the first parameters)
    :param parameter_model_weights: pretrained weights for the parameter model
    :return: full model and parameter model
    
    """
    
    audio_time = tfkl.Input(shape=(sample_length,), name="audio_time")
    logmelgram = layers.LogMelgramLayer(1024, 256, 128, sample_rate, 0.0, sample_rate//2, 1e-6)
    x = logmelgram(audio_time)

    x = tfkl.Conv2D(32, kernel_size=2, activation='relu')(x)
    x = tfkl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tfkl.BatchNormalization()(x)

    x = tfkl.Flatten()(x)
    x = tfkl.Dense(32, activation='relu')(x)
    x = tfkl.Dropout(.1, noise_shape=(n_processors, 32))(x)

    x = tfkl.Dense(n_parameters, activation="sigmoid")(x)
    params = tf.math.scalar_mul(100.0, x)    

    parameter_model = tfk.models.Model(audio_time, params, name="parameter_model")
    parameters = parameter_model(audio_time)
    if parameter_model_weights is not None:
        parameter_model.load_weights(parameter_model_weights)

    vstlayer = layers.VSTProcessor(vst_path, sample_rate, n_processors, epsilon, parameter_map, non_learnable_parameters)
    output = vstlayer([audio_time, parameters])

    full_model = tfk.models.Model(audio_time, output, name="full_model")

    return full_model, parameter_model, vstlayer

def get_models(sample_length, sample_rate, n_parameters, n_processors, vst_path, epsilon, 
    parameter_map, non_learnable_parameters, parameter_model_weights=None):
    """
    Get full model and parameter model

    :param sample_length: length of the input audio
    :param sample_rate: desired sample rate of neural network
    :param n_parameters: number of parameters to be estimated
    :param n_processors: number of processors to be used
    :param vst_path: path to the VST plugin
    :param epsilon: epsilon for the SPSA
    :param n_delay_lines: number of delay lines
    :param true_spsa: use true SPSA or not
    :param linked_EQs: use linked EQs or not
    :param parameter_offset: offset for the parameters (to ignore the first parameters)
    :param parameter_model_weights: pretrained weights for the parameter model
    :return: full model and parameter model
    
    """

    audio_time = tfkl.Input(shape=(sample_length,), name="audio_time")
    print('input shape:', audio_time.shape)
    logmelgram = layers.LogMelgramLayer(frame_length=1024,
                                        num_fft=1024,
                                        hop_length=256,
                                        num_mels=128,
                                        sample_rate=sample_rate,
                                        f_min=0.0,
                                        f_max=sample_rate // 2,
                                        eps=1e-6,
                                        norm=False, name='logMelgram')
    x = logmelgram(audio_time)
    x = tfkl.Conv2D(128, kernel_size=2, activation='relu')(x)
    x = tfkl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tfkl.BatchNormalization()(x)

    x = tfkl.Conv2D(32, kernel_size=2, activation='relu')(x)
    x = tfkl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tfkl.BatchNormalization()(x)

    x = tfkl.Conv2D(16, kernel_size=2, activation='relu')(x)
    x = tfkl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tfkl.BatchNormalization()(x)

    x = tfkl.Conv2D(32, kernel_size=2, activation='relu')(x)
    x = tfkl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tfkl.BatchNormalization()(x)

    x = tfkl.Flatten()(x)
    x = tfkl.Dense(32, activation='relu')(x)
    x = tfkl.Dropout(.05, noise_shape=(n_processors, 32))(x)

    params = tfkl.Dense(n_parameters, activation="sigmoid")(x)   

    parameter_model = tfk.models.Model(audio_time, params, name="parameter_model")
    parameters = parameter_model(audio_time)
    if parameter_model_weights is not None:
        parameter_model.load_weights(parameter_model_weights)

    vstlayer = layers.VSTProcessor(vst_path, sample_rate, n_processors, epsilon, parameter_map, non_learnable_parameters)
    output = vstlayer([audio_time, parameters])

    full_model = tfk.models.Model(audio_time, output, name="full_model")

    return full_model, parameter_model, vstlayer

def get_mobilenet_models(sample_length, sample_rate, n_parameters, n_processors, vst_path, epsilon, 
    parameter_map, non_learnable_parameters, parameter_model_weights=None):

    audio_time = tfkl.Input(shape=(sample_length,), name="audio_time")
    logmelgram = layers.LogMelgramLayer(frame_length=1024,
                                        num_fft=1024,
                                        hop_length=256,
                                        num_mels=128,
                                        sample_rate=sample_rate,
                                        f_min=0.0,
                                        f_max=sample_rate // 2,
                                        eps=1e-6,
                                        norm=False, name='logMelgram')

    x = logmelgram(audio_time)
    x = tfkl.BatchNormalization(name="input_norm")(x)
    encoder_model = tfk.applications.MobileNetV2(input_shape=(x.shape[1], x.shape[2], x.shape[3]), alpha=1.0,
                                        include_top=True, weights=None, input_tensor=None, pooling=None,
                                        classes=np.sum(n_parameters).item(), classifier_activation="sigmoid")

    hidden = encoder_model(x)

    parameter_model = tfk.models.Model(audio_time, hidden, name="parameter_model")

    parameters = parameter_model(audio_time)
    
    if parameter_model_weights is not None:
        parameter_model.load_weights(parameter_model_weights)

    vstlayer = layers.VSTProcessor(vst_path, sample_rate, n_processors, 
                                   epsilon, parameter_map, non_learnable_parameters)
    output = vstlayer([audio_time, parameters])

    full_model = tfk.models.Model(audio_time, output, name="full_model")

    return full_model, parameter_model, vstlayer

def pretrain_parameter_model(input_audio, target_parameters):
    sample_rate = 8000
    _, parameter_model = get_models(input_audio.shape[1], sample_rate, target_parameters.shape[1], 1, "", 0.0)
    optimizer = tfk.optimizers.Adam(learning_rate=0.0001) 
    parameter_model.compile(optimizer=optimizer, loss='mae', run_eagerly=True)
    lr_callback = tfk.callbacks.ReduceLROnPlateau(monitor='loss',
                              factor=0.2,
                              patience=100,
                              cooldown=1,
                              verbose=1,
                              mode='auto',
                              min_lr=0.000001)
    history = parameter_model.fit(input_audio, target_parameters, verbose=1, epochs=1000, batch_size=1,
         callbacks=[lr_callback])
    return parameter_model, history