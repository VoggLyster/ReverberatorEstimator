k = {}

k['sample_rate'] = 48000
k['sample_length'] = 96000
k['epochs'] = 3000
k['n_processors'] = 4
k['steps_per_epoch'] = 1
k['epsilon'] = 0.01
k['learning_rate'] = 1e-5

k['dry_audio_path'] = './Dataset/Dry/0.wav'
k['wet_audio_path'] = './Dataset/Wet/AbletonReverb/0.wav'

k['vst_path'] = './Reverberator_16ch.vst3'

k['time_loss_weight'] = 0.0
k['spectral_loss_weight'] = 1.0
k['envelope_loss_weight'] = 100.0
k['echo_density_loss_weight'] = 1.0

k['use_multiscale'] = True

k['n_parameters'] = []
k['parameter_map'] = []
k['non_trainable_parameters'] = []

parameters = [0,1,2,
              3,4,5,6,7,8,9,10,11,12,
              13,14,15,16,17,18,19,20,21,22,
              23,24,27,28,31,32,35,36,39,40,43,44,47,48,51,52,55,56,59,60,63,64,67,68,71,72,75,76,79,80,83,84]
non_trainable_parameters = {87:1}

nt_mod_keys = [25,26,29,30,33,34,37,38,41,42,45,46,49,50,53,54,57,58,61,62,65,66,69,70,73,74,77,78,81,82,85,86]
nt_mod_vals = [0.0] * len(nt_mod_keys)
non_trainable_parameters.update(dict(zip(nt_mod_keys, nt_mod_vals)))

n_parameters = len(parameters)
parameter_map = {}
for i, par_idx in enumerate(parameters):
    parameter_map[i] = par_idx
k['n_parameters'] = n_parameters
k['parameter_map'] = parameter_map
k['non_trainable_parameters'] = non_trainable_parameters

k['checkpoint_path'] = 'checkpoints_ableton/cp.ckpt'
k['pretrained_weights'] = 'pretrained_model_ableton/ckpt'

