########################################################################################################################
#                                          N-HANS speech denoiser: apply                                               #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#   Description:      Apply N-HANS trained denoiser for audio.                          .                              #
#   Authors:          Shuo Liu, Gil Keren, Bjoern Schuller                                                             #
#   Affiliation:      ZD.B Chair of Embedded Intelligence for Health Care and Wellbeing, University of Augsburg (UAU)  #
#   Version:          1.0                                                                                              #
#   Last Update:      Nov. 14, 2019                                                                                    #
#   Dependence Files: xxx                                                                                              #
#   Contact:          shuo.liu@informatik.uni-augburg.de                                                               #
########################################################################################################################

from __future__ import division, absolute_import, print_function
import numpy as np
import tensorflow as tf
import math
import os
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from main import model


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('trained_model_dir', './trained_model', '')
tf.app.flags.DEFINE_string('trained_model_name', '81448_0-1000000', '')

## denosing single sample
# tf.app.flags.DEFINE_string('input', './audio_examples/exp2_noisy.wav', '')
# tf.app.flags.DEFINE_string('neg', './audio_examples/exp2_noise.wav', '')
# tf.app.flags.DEFINE_string('pos', './audio_examples/Silent.wav', '')
# tf.app.flags.DEFINE_string('output', './audio_examples/exp2_denoised.wav', '')

## Selective noise compression example
tf.app.flags.DEFINE_string('input', './audio_examples/exp1_noisy.wav', '')
tf.app.flags.DEFINE_string('neg', './audio_examples/exp1_-noise.wav', '')
tf.app.flags.DEFINE_string('pos', './audio_examples/exp1_+noise.wav', '')
tf.app.flags.DEFINE_string('output', './audio_examples/exp1_denoised.wav', '')

## denoising multiple samples
# tf.app.flags.DEFINE_string('input', './wav_folder/noisy', '')
# tf.app.flags.DEFINE_string('neg', './wav_folder/noise', '')
# tf.app.flags.DEFINE_string('pos', './audio_examples/Silent.wav', '')
# tf.app.flags.DEFINE_string('output', './wav_folder/denoised', '')


tf.app.flags.DEFINE_float('compensate', 0, '')
tf.app.flags.DEFINE_boolean('ac', False, '')

Noise_Win = 200
Mix_Win = 35


#############################################     Assistence Functions     #############################################

def read_wav(in_path):
  rate, samples = wavread(in_path)
  assert rate == 16000
  # assert samples.dtype == 'int16'
  if len(samples.shape) > 1:
    samples = samples.mean(axis=1)
  assert len(samples.shape) == 1
  return samples


def domixing(cleansamples, noisepossamples, noisenegsamples, snr_pos, snr_neg):
  # Repeat the noise if it is shorter than the speech, or shorter it if it's longer
  nse_pos = noisepossamples
  nse_neg = noisenegsamples
  while len(cleansamples) - len(nse_pos) > 0: # Make noise longer
    diff = len(cleansamples) - len(nse_pos)
    nse_pos = np.concatenate([nse_pos, noisepossamples[:diff]], axis=0)

  while len(cleansamples) - len(nse_neg) > 0:  # Make noise longer
      diff = len(cleansamples) - len(nse_neg)
      nse_neg = np.concatenate([nse_neg, noisenegsamples[:diff]], axis=0)

  if len(cleansamples) - len(noisepossamples) < 0:  # Make noise shorter
      nse_pos = noisepossamples[:len(cleansamples)]

  if len(cleansamples) - len(noisenegsamples) < 0:  # Make noise shorter
      nse_neg = noisenegsamples[:len(cleansamples)]

  sig = cleansamples

  # Power of signal and noise
  psignal = sum(abs(sig) * abs(sig)) / sig.shape[0]
  pnoise_pos = sum(abs(nse_pos) * abs(nse_pos)) / nse_pos.shape[0]
  pnoise_neg = sum(abs(nse_neg) * abs(nse_neg)) / nse_neg.shape[0]

  # Compute scale factor
  if pnoise_pos == 0:
    K_pos = 1
  else:
    K_pos = (psignal / pnoise_pos) * pow(10, -snr_pos / 10.0)
    K_pos = np.sqrt(K_pos)

  if pnoise_neg == 0:
    K_neg = 1
  else:
    K_neg = (psignal / pnoise_neg) * pow(10, -snr_neg / 10.0)
    K_neg = np.sqrt(K_neg)



  # Mix
  noise_pos_scaled = K_pos * nse_pos                                # Scale the noise
  noise_neg_scaled = K_neg * nse_neg                                # Scale the noise
  mixed = sig + noise_pos_scaled + noise_neg_scaled                            # Mix
  mixed = mixed / (max(abs(mixed))+0.000001)            # Normalize
  target = sig + noise_pos_scaled
  target = target / (max(abs(mixed)) + 0.000001)        # Normalize
  noise_pos_signal = noise_pos_scaled/(max(abs(mixed))+0.000001)
  noise_neg_signal = noise_neg_scaled/(max(abs(mixed))+0.000001)

  return mixed, target, K_pos, K_neg, noise_pos_signal, noise_neg_signal


def combine_signals(cleanpath, noisepospath, noisenegpath):
  try:
    # Read Wavs
    cleansamples = read_wav(cleanpath)
    noisepossamples = read_wav(noisepospath)
    noisenegsamples = read_wav(noisenegpath)

    # Normalize
    cleansamples = cleansamples / (max(abs(cleansamples))+0.000001)
    noisepossamples = noisepossamples / (max(abs(noisepossamples))+0.000001)
    noisenegsamples = noisenegsamples / (max(abs(noisenegsamples))+0.000001)
    cleansamples = cleansamples.astype(np.float32)
    noisepossamples = noisepossamples.astype(np.float32)
    noisenegsamples = noisenegsamples.astype(np.float32)

    # Cut the end to have an exact number of frames
    if (len(cleansamples) - 400) % 160 != 0:
        cleansamples = cleansamples[:-((len(cleansamples) - 400) % 160)]

    # Choose SNR
    SNRs = [-3, 0, 3, 5, 8]


    snr_pos = SNRs[1]
    snr_neg = SNRs[1]

    mixed, target, Kpos, Kneg, noisepossignal, noisenegsignal = domixing(cleansamples, noisepossamples, noisenegsamples, snr_pos, snr_neg)
    return noisepossignal, noisenegsignal, mixed, np.array(snr_pos, dtype=np.int32), np.array(snr_neg, dtype=np.int32)

  except:
    print('error in threads')
    print(cleanpath, noisepospath, noisenegpath)


def handle_signals(mixedpath, noisepospath, noisenegpath):
  try:
    # Read Wavs
    mixedsamples = read_wav(mixedpath)
    noisepossamples = read_wav(noisepospath)
    noisenegsamples = read_wav(noisenegpath)

    # Normalize
    mixedsamples = mixedsamples / (max(abs(mixedsamples))+0.000001)
    noisepossamples = noisepossamples / (max(abs(noisepossamples))+0.000001)
    noisenegsamples = noisenegsamples / (max(abs(noisenegsamples))+0.000001)
    mixedsamples = mixedsamples.astype(np.float32)
    noisepossamples = noisepossamples.astype(np.float32)
    noisenegsamples = noisenegsamples.astype(np.float32)

    # Cut the end to have an exact number of frames
    if (len(mixedsamples) - 400) % 160 != 0:
        mixedsamples = mixedsamples[:-((len(mixedsamples) - 400) % 160)]

    nsepos = noisepossamples
    nseneg = noisenegsamples
    while len(mixedsamples) - len(nsepos) > 0:  # Make noise longer
        diff = len(mixedsamples) - len(nsepos)
        nsepos = np.concatenate([nsepos, noisepossamples[:diff]], axis=0)

    while len(mixedsamples) - len(nseneg) > 0:  # Make noise longer
        diff = len(mixedsamples) - len(nseneg)
        nseneg = np.concatenate([nseneg, noisenegsamples[:diff]], axis=0)

    if len(mixedsamples) - len(noisepossamples) < 0:  # Make noise shorter
        nsepos = noisepossamples[:len(mixedsamples)]
    if len(mixedsamples) - len(noisenegsamples) < 0:  # Make noise shorter
        nseneg = noisenegsamples[:len(mixedsamples)]

    noisepossamples = nsepos
    noisenegsamples = nseneg

    return noisepossamples, noisenegsamples, mixedsamples

  except:
    print('error in threads')
    print(mixedpath, noisepospath, noisenegpath)


def pad_1D_for_windowing(tensor, length):
    len_before = ((length + 1) // 2) - 1
    len_after = length // 2
    return tf.pad(tensor, [[len_before, len_after],[0,0]])


def strided_crop(tensor, length, stride):
  # we assume that we have a length dimension and a feature dimension
  assert len(tensor.shape) == 2
  n_features = int(tensor.shape[1])
  padded = pad_1D_for_windowing(tensor, length)
  windows = tf.extract_image_patches(tf.expand_dims(tf.expand_dims(padded, axis=0), axis=3),
                                           ksizes=[1, length, n_features, 1],
                                           strides=[1, stride, n_features, 1],
                                           rates=[1, 1, 1, 1],
                                           padding='VALID')
  return tf.reshape(windows, [-1, length, n_features])


def recover_samples_from_spectrum(logspectrum_stft, spectrum_phase, save_to=None):
    abs_spectrum = np.exp(logspectrum_stft)
    spectrum = abs_spectrum * (np.exp(1j * spectrum_phase))

    istft_graph = tf.Graph()
    with istft_graph.as_default():
        stft_ph = tf.placeholder(tf.complex64, shape=(None, 201))
        samples = tf.contrib.signal.inverse_stft(stft_ph, 400, 160, 400)
        istft_sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        samples_ = istft_sess.run(samples, feed_dict={stft_ph: spectrum})
        if save_to:
            wavwrite(save_to, 16000, samples_)

    return samples_


################################################     Apply N-HANS speech denoiser     ##################################

def apply_demo(speechpath, pospath, negpath, save_to):

    speechlist = []
    speechlist.append(speechpath)
    noiselist = []
    noiselist.append(pospath)
    noiselist.append(negpath)

    # data processing
    g = tf.Graph()
    with g.as_default():
        with tf.device('/cpu:0'):
            speechlist = tf.constant(np.array(speechlist))
            noiselist = tf.constant(np.array(noiselist))

            seeds = []
            seeds.append(speechlist)
            seeds.append(noiselist)

            speechpath_ph = speechlist[0]
            pospath_ph = noiselist[0]
            negpath_ph = noiselist[1]

            noise_pos_wav, noise_neg_wav, mix_wav, snr_pos, snr_neg = tf.py_func(combine_signals,
                                                                                 [speechpath_ph, pospath_ph,
                                                                                  negpath_ph],
                                                                                 [tf.float32,
                                                                                  tf.float32, tf.float32,
                                                                                  tf.int32, tf.int32])

            noise_pos_wav, noise_neg_wav, mix_wav = [tf.reshape(x, [-1]) for x in (noise_pos_wav, noise_neg_wav, mix_wav)]

            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

            mix_stft, pos_stft, neg_stft = [tf.contrib.signal.stft(wav, 400, 160, fft_length=400) for wav in [mix_wav, noise_pos_wav, noise_neg_wav]]

            mix_spectrum, pos_spectrum, neg_spectrum = [tf.log(tf.abs(wav_stft) + 1e-5) for wav_stft in [mix_stft, pos_stft, neg_stft]]

            mix_phase = tf.angle(mix_stft)


            # crop data
            mix_spectra = strided_crop(mix_spectrum[Noise_Win:], Mix_Win, 1)

            # postive noise & negative noise
            pos_spectrum = pos_spectrum[:Noise_Win]
            pos_spectrum = tf.reshape(pos_spectrum, [Noise_Win, pos_spectrum.shape[1].value])
            pos_spectra = tf.tile(tf.expand_dims(pos_spectrum, 0), [tf.shape(mix_spectra)[0], 1, 1])

            neg_spectrum = neg_spectrum[:Noise_Win]
            neg_spectrum = tf.reshape(neg_spectrum, [Noise_Win, neg_spectrum.shape[1].value])
            neg_spectra = tf.tile(tf.expand_dims(neg_spectrum, 0), [tf.shape(mix_spectra)[0], 1, 1])

            mixedphs = strided_crop(mix_phase[Noise_Win:], 1, 1)

            # get data
            # print('--------------------------------')
            mix_spectra_, pos_spectra_, neg_spectra_ = sess.run([mix_spectra, pos_spectra, neg_spectra])
            mixedphs_ = sess.run(mixedphs, feed_dict={speechpath_ph: speechpath,
                                                      pospath_ph: pospath,
                                                      negpath_ph: negpath})


    mb = 100
    batches = int(math.ceil(len(mix_spectra_) / float(mb)))
    denoised = []
    mix_centers = []

    ## denoiseing
    # graph
    eg = tf.Graph()
    with eg.as_default():
        esess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        with tf.device('/gpu:0'):
            in_ph = [tf.placeholder(dtype=tf.float32, shape=[None, 1, 201], name='targetph'),
                  tf.placeholder(dtype=tf.float32, shape=[None, 35, 201], name='mixedph'),
                  tf.placeholder(dtype=tf.float32, shape=[None, 1, 201], name='mixedphaseph'),
                  tf.placeholder(dtype=tf.float32, shape=[None, 1, 201], name='targetphaseph'),
                  tf.placeholder(dtype=tf.float32, shape=[None, 35, 201], name='posph'),
                  tf.placeholder(dtype=tf.float32, shape=[None, 1, 201], name='posphaseph'),
                  tf.placeholder(dtype=tf.float32, shape=[None, 35, 201], name='negph'),
                  tf.placeholder(dtype=tf.float32, shape=[None, 1, 201], name='negphaseph'),
                  tf.placeholder(dtype=tf.float32, shape=[None, 200, 201], name='noiseposcontextph'),
                  tf.placeholder(dtype=tf.float32, shape=[None, 200, 201], name='noisenegcontextph'),
                  tf.placeholder(dtype=tf.int32, shape=[None], name='locationph'),
                  tf.placeholder(dtype=tf.string, shape=[None], name='targetpathph'),
                  tf.placeholder(dtype=tf.string, shape=[None], name='noisepospathph'),
                  tf.placeholder(dtype=tf.string, shape=[None], name='noisenegpathph'),
                  tf.placeholder(dtype=tf.int32, shape=[None], name='snrposph'),
                  tf.placeholder(dtype=tf.int32, shape=[None], name='snrnegph'),
                  ]

            _, _, outputs = model(in_ph, False)
        esaver = tf.train.Saver(tf.global_variables())

    # load in trained model
    checkpoints_dir = FLAGS.trained_model_dir
    model_name = FLAGS.trained_model_name
    esaver = tf.train.import_meta_graph(os.path.join(checkpoints_dir, model_name + '.meta'))
    esaver.restore(esess, os.path.join(checkpoints_dir, model_name))

    mixed_tensor = eg.get_tensor_by_name('mixedph:0')
    pos_noise_tensor = eg.get_tensor_by_name('noiseposcontextph:0')
    neg_noise_tensor = eg.get_tensor_by_name('noisenegcontextph:0')
    denoised_tensor = eg.get_tensor_by_name('add_72:0')

    # nn processing
    for i in range(batches):
        batch_mix_spectrum, batch_pos_spectrum, batch_neg_spectrum = [spectra[i*mb:(i+1)*mb] for spectra in [mix_spectra_, pos_spectra_, neg_spectra_]]

        batch_denoised_ = esess.run(denoised_tensor, feed_dict={mixed_tensor: batch_mix_spectrum,
                                                                pos_noise_tensor: batch_pos_spectrum,
                                                                neg_noise_tensor: batch_neg_spectrum})

        denoised.append(batch_denoised_)
        mix_center = batch_mix_spectrum[:, 17, :]
        mix_centers.append(mix_center)

    # reconstruction
    denoised = np.concatenate(denoised, axis=0)
    mix_centers = np.concatenate(mix_centers, axis=0)

    recover_samples_from_spectrum(denoised, mixedphs_[:, 0, :], save_to)
    mix_save_to = save_to[:-15] + 'mixed_demo.wav'
    recover_samples_from_spectrum(mix_centers, mixedphs_[:, 0, :], mix_save_to)


def apply_snc(mixedpath, pospath, negpath, save_to):
    mixedlist = []
    mixedlist.append(mixedpath)
    posnoiselist = []
    negnoiselist = []
    posnoiselist.append(pospath)
    negnoiselist.append(negpath)

    # data processing
    g = tf.Graph()
    with g.as_default():
        with tf.device('/cpu:0'):
            mixedlist = tf.constant(np.array(mixedlist))
            posnoiselist = tf.constant(np.array(posnoiselist))
            negnoiselist = tf.constant(np.array(negnoiselist))

            mixedpath_ph = mixedlist[0]
            pospath_ph = posnoiselist[0]
            negpath_ph = negnoiselist[0]

            noise_pos_wav, noise_neg_wav, mix_wav = tf.py_func(handle_signals,
                                                               [mixedpath_ph, pospath_ph, negpath_ph],
                                                               [tf.float32, tf.float32, tf.float32])

            noise_pos_wav, noise_neg_wav, mix_wav = [tf.reshape(x, [-1]) for x in
                                                     (noise_pos_wav, noise_neg_wav, mix_wav)]

            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

            mix_stft, pos_stft, neg_stft = [tf.contrib.signal.stft(wav, 400, 160, fft_length=400) for wav in
                                            [mix_wav, noise_pos_wav, noise_neg_wav]]

            mix_spectrum, pos_spectrum, neg_spectrum = [tf.log(tf.abs(wav_stft) + 1e-5) for wav_stft in
                                                        [mix_stft, pos_stft, neg_stft]]
            mix_phase = tf.angle(mix_stft)


            # crop data
            mix_spectra = strided_crop(mix_spectrum, Mix_Win, 1)

            # postive noise & negative noise
            pos_spectrum = pos_spectrum[:Noise_Win]
            pos_spectrum = tf.reshape(pos_spectrum, [Noise_Win, pos_spectrum.shape[1].value])
            pos_spectra = tf.tile(tf.expand_dims(pos_spectrum, 0), [tf.shape(mix_spectra)[0], 1, 1])

            neg_spectrum = neg_spectrum[:Noise_Win]
            neg_spectrum = tf.reshape(neg_spectrum, [Noise_Win, neg_spectrum.shape[1].value])
            neg_spectra = tf.tile(tf.expand_dims(neg_spectrum, 0), [tf.shape(mix_spectra)[0], 1, 1])

            mixedphs = strided_crop(mix_phase, 1, 1)

            # get data
            # print('------------------------------------------------------------------------------------')
            mix_spectra_, pos_spectra_, neg_spectra_ = sess.run([mix_spectra, pos_spectra, neg_spectra])
            mixedphs_ = sess.run(mixedphs, feed_dict={mixedpath_ph: mixedpath,
                                                      pospath_ph: pospath,
                                                      negpath_ph: negpath})

    mb = 100
    batches = int(math.ceil(len(mix_spectra_) / float(mb)))
    denoised = []
    mix_centers = []

    ## denoiseing
    # graph
    eg = tf.Graph()
    with eg.as_default():
        esess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        with tf.device('/gpu:0'):
            in_ph = [tf.placeholder(dtype=tf.float32, shape=[None, 1, 201], name='targetph'),
                     tf.placeholder(dtype=tf.float32, shape=[None, 35, 201], name='mixedph'),
                     tf.placeholder(dtype=tf.float32, shape=[None, 1, 201], name='mixedphaseph'),
                     tf.placeholder(dtype=tf.float32, shape=[None, 1, 201], name='targetphaseph'),
                     tf.placeholder(dtype=tf.float32, shape=[None, 35, 201], name='posph'),
                     tf.placeholder(dtype=tf.float32, shape=[None, 1, 201], name='posphaseph'),
                     tf.placeholder(dtype=tf.float32, shape=[None, 35, 201], name='negph'),
                     tf.placeholder(dtype=tf.float32, shape=[None, 1, 201], name='negphaseph'),
                     tf.placeholder(dtype=tf.float32, shape=[None, 200, 201], name='noiseposcontextph'),
                     tf.placeholder(dtype=tf.float32, shape=[None, 200, 201], name='noisenegcontextph'),
                     tf.placeholder(dtype=tf.int32, shape=[None], name='locationph'),
                     tf.placeholder(dtype=tf.string, shape=[None], name='targetpathph'),
                     tf.placeholder(dtype=tf.string, shape=[None], name='noisepospathph'),
                     tf.placeholder(dtype=tf.string, shape=[None], name='noisenegpathph'),
                     tf.placeholder(dtype=tf.int32, shape=[None], name='snrposph'),
                     tf.placeholder(dtype=tf.int32, shape=[None], name='snrnegph'),
                     ]
            _, _, outputs = model(in_ph, False)
        esaver = tf.train.Saver(tf.global_variables())

    # load in trained model
    checkpoints_dir = FLAGS.trained_model_dir
    model_name = FLAGS.trained_model_name
    esaver = tf.train.import_meta_graph(os.path.join(checkpoints_dir, model_name + '.meta'))
    esaver.restore(esess, os.path.join(checkpoints_dir, model_name))

    mixed_tensor = eg.get_tensor_by_name('mixedph:0')
    pos_noise_tensor = eg.get_tensor_by_name('noiseposcontextph:0')
    neg_noise_tensor = eg.get_tensor_by_name('noisenegcontextph:0')
    denoised_tensor = eg.get_tensor_by_name('add_72:0')

    # nn processing
    for i in range(batches):
        batch_mix_spectrum, batch_pos_spectrum, batch_neg_spectrum = [spectra[i * mb:(i + 1) * mb] for spectra in
                                                                      [mix_spectra_, pos_spectra_, neg_spectra_]]

        batch_denoised_ = esess.run(denoised_tensor, feed_dict={mixed_tensor: batch_mix_spectrum,
                                                                pos_noise_tensor: batch_pos_spectrum,
                                                                neg_noise_tensor: batch_neg_spectrum})

        denoised.append(batch_denoised_)
        mix_center = batch_mix_spectrum[:, 17, :]
        mix_centers.append(mix_center)

    # reconstruction
    denoised = np.concatenate(denoised, axis=0)
    mix_centers = np.concatenate(mix_centers, axis=0)

    denoised_samples = recover_samples_from_spectrum(denoised, mixedphs_[:, 0, :], save_to)
    mixed_samples = recover_samples_from_spectrum(mix_centers, mixedphs_[:, 0, :])
    removed_samples = mixed_samples - denoised_samples

    snr_est = np.mean(np.square(denoised_samples)) / np.mean(np.square(removed_samples))

    if not FLAGS.ac:
        compensation_factor = FLAGS.compensate
    else:
        compensation_factor = snr_est/20

    denoised_samples += removed_samples * compensation_factor

    verbose = 0
    if verbose:
        print(snr_est)
        print('---------------------------------------------------------------------------------------')
        save_to = save_to if len(save_to.split('/')[-1]) == 0 else save_to[:-len(save_to.split('/')[-1])]
        mix_save_to = save_to + 'mixed_processed.wav'
        mixed_samples = recover_samples_from_spectrum(mix_centers, mixedphs_[:, 0, :], mix_save_to)
        removed_save_to = save_to + 'removed.wav'
        wavwrite(removed_save_to, 16000, removed_samples)
        compensated_save_to = save_to + 'compensated.wav'
        compensated_samples = denoised_samples + removed_samples * compensation_factor
        wavwrite(compensated_save_to, 16000, compensated_samples)

    # snr = S/N   snr_est = S-e / N+e
    # e is distortion, need to be compensate. For a definite snr, Lower snr_est requires more compensation.

def apply_denoiser(mixedpath, negpath, save_to):
    dir = './audio_examples/'
    pospath = dir + 'Silent.wav'
    apply_snc(mixedpath, pospath, negpath, save_to)


if __name__=='__main__':


    mixedpath = FLAGS.input
    negpath = FLAGS.neg
    pospath = FLAGS.pos
    save_to = FLAGS.output


    # # To test denoising, uncommet the folloing:
    # if os.path.isdir(mixedpath) and os.path.isdir(negpath) and os.path.isdir(save_to):
    #     print('##################################')
    #     print(' Applying N-HANS denoising system ')
    #     print('##################################')
    #
    #     for file in os.listdir(mixedpath):
    #         mixedwav = os.path.join(mixedpath, file)
    #         negwav = os.path.join(negpath, file)
    #         savewav = os.path.join(save_to, file)
    #         print('Processing:')
    #         print('  Input : {}'.format(mixedwav))
    #         print('  -rec  : {}'.format(negwav))
    #         apply_denoiser(mixedwav, negwav, savewav)
    #         print('Processed result:')
    #         print('  Output: {}'.format(savewav))
    #         print('------------------------------------')
    #         print('')
    #
    #
    # elif mixedpath.endswith('.wav') and negpath.endswith('.wav') and save_to.endswith('.wav'):
    #     print('##################################')
    #     print(' Applying N-HANS denoising system ')
    #     print('##################################')
    #
    #     print('Processing:')
    #     print('  Input : {}'.format(mixedpath))
    #     print('  -rec  : {}'.format(negpath))
    #     apply_denoiser(mixedpath, negpath, save_to)
    #     print('Processed result:')
    #     print('  Output: {}'.format(save_to))
    #     print('----------------------------------')
    #     print('')
    #
    # else:
    #     print('Please give the .wav file path or the path to folders containing .wav files')
    #


    ## To test selective noise suppresion, uncommet the folloing:
    if os.path.isdir(mixedpath) and os.path.isdir(negpath) and os.path.isdir(pospath) and os.path.isdir(save_to):
        print('####################################################')
        print(' Applying N-HANS selective noise suppression system ')
        print('####################################################')
        for file in os.listdir(mixedpath):
            mixedwav = os.path.join(mixedpath, file)
            negwav = os.path.join(negpath, file)
            poswav = os.path.join(pospath, file)
            savewav = os.path.join(save_to, file)
            print('Processing:')
            print('  Input : {}'.format(mixedwav))
            print('  +rec  : {}'.format(poswav))
            print('  -rec  : {}'.format(negwav))
            apply_snc(mixedwav, poswav, negwav, savewav)
            print('Processed result:')
            print('  Output: {}'.format(savewav))
            print('----------------------------------')
            print('')

    elif mixedpath.endswith('.wav') and pospath.endswith('.wav') and negpath.endswith('.wav') and save_to.endswith('.wav'):
        print('####################################################')
        print(' Applying N-HANS selective noise suppression system ')
        print('####################################################')
        print('Processing:')
        print('  Input : {}'.format(mixedpath))
        print('  +rec  : {}'.format(pospath))
        print('  -rec  : {}'.format(negpath))
        apply_snc(mixedpath, pospath, negpath, save_to)
        print('Processed result:')
        print('  Output: {}'.format(save_to))
        print('----------------------------------')
        print('')

    else:
        print('Please give the .wav file path or the path to folders containing .wav files')





