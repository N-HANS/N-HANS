########################################################################################################################
#                                          N-HANS speech separator: apply                                              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#   Description:      Apply N-HANS trained separator for audio.                                                        #
#   Authors:          Shuo Liu, Gil Keren, Bjoern Schuller                                                             #
#   Affiliation:      ZD.B Chair of Embedded Intelligence for Health Care and Wellbeing, University of Augsburg (UAU)  #
#   Version:          1.0                                                                                              #
#   Last Update:      Nov. 14, 2019                                                                                    #
#   Dependence Files: main.py                                                                                          #
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
tf.app.flags.DEFINE_string('trained_model_name', '81457_2-545000', '')

## to test single sample
# tf.app.flags.DEFINE_string('input', './audio_examples/mixed.wav', '')
# tf.app.flags.DEFINE_string('neg', './audio_examples/noise_speaker.wav', '')
# tf.app.flags.DEFINE_string('pos', './audio_examples/target_speaker.wav', '')
# tf.app.flags.DEFINE_string('output', './audio_examples/denoised.wav', '')

## to test multiple samples
tf.app.flags.DEFINE_string('input', './wav_folder/mixed', '')
tf.app.flags.DEFINE_string('neg', './wav_folder/noise_speaker', '')
tf.app.flags.DEFINE_string('pos', './wav_folder/target_speaker', '')
tf.app.flags.DEFINE_string('output', './wav_folder/denoised', '')

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


def domixing(cleansamples, noisesamples, snr):
  # Repeat the noise if it is shorter than the speech, or shorter it if it's longer
  nse = noisesamples
  while len(cleansamples) - len(nse) > 0: # Make noise longer
    diff = len(cleansamples) - len(nse)
    nse = np.concatenate([nse, noisesamples[:diff]], axis=0)
  if len(cleansamples) - len(noisesamples) < 0: # Make noise shorter
    nse = noisesamples[:len(cleansamples)]
  sig = cleansamples

  # Power of signal and noise
  psignal = sum(abs(sig) * abs(sig)) / sig.shape[0]
  pnoise  = sum(abs(nse) * abs(nse)) / nse.shape[0]

  # Compute scale factor
  if pnoise == 0:
    K = 1
  else:
    K = (psignal / pnoise) * pow(10, -snr / 10.0)
  K = np.sqrt(K)

  # Mix
  noise_scaled = K * nse                                # Scale the noise
  mixed = sig + noise_scaled                            # Mix
  mixed = mixed / (max(abs(mixed))+0.000001)            # Normalize
  return mixed, K


def combine_signals(cleanpath, noisepath):
  try:
    # Read Wavs
    cleansamples = read_wav(cleanpath)
    noisesamples = read_wav(noisepath)

    # Normalize
    cleansamples = cleansamples / (max(abs(cleansamples))+0.000001)
    noisesamples = noisesamples / (max(abs(noisesamples))+0.000001)
    cleansamples = cleansamples.astype(np.float32)
    noisesamples = noisesamples.astype(np.float32)

    # Cut the end to have an exact number of frames
    cleansamples = cleansamples[:-((len(cleansamples) - 400) % 160)]

    # Choose SNR
    SNRs = [-5, -3, -1, 0, 1, 3, 5]
    # The noise is the beginning of the file, and mix the rest
    snr = 0
    mixed, K = domixing(cleansamples, noisesamples, snr)
    return cleansamples, noisesamples * K, mixed, np.array(snr, dtype=np.int32)

  except:
    print('error in threads')
    print(cleanpath, noisepath)


def handle_signals(mixedpath, cleanpath, noisepath):
  try:
    # Read Wavs
    mixedsamples = read_wav(mixedpath)
    cleansamples = read_wav(cleanpath)
    noisesamples = read_wav(noisepath)

    # Normalize
    mixedsamples = mixedsamples / (max(abs(mixedsamples))+0.000001)
    cleansamples = cleansamples / (max(abs(cleansamples))+0.000001)
    noisesamples = noisesamples / (max(abs(noisesamples))+0.000001)
    mixedsamples = mixedsamples.astype(np.float32)
    cleansamples = cleansamples.astype(np.float32)
    noisesamples = noisesamples.astype(np.float32)

    # Cut the end to have an exact number of frames
    if (len(mixedsamples) - 400) % 160 != 0:
        mixedsamples = mixedsamples[:-((len(mixedsamples) - 400) % 160)]

    nse = noisesamples
    cln = cleansamples
    while len(mixedsamples) - len(nse) > 0:  # Make noise longer
        diff = len(mixedsamples) - len(nse)
        nse = np.concatenate([nse, noisesamples[:diff]], axis=0)

    while len(mixedsamples) - len(cln) > 0:  # Make noise longer
        diff = len(mixedsamples) - len(cln)
        cln = np.concatenate([cln, noisesamples[:diff]], axis=0)

    if len(mixedsamples) - len(noisesamples) < 0:  # Make noise shorter
        nse = noisesamples[:len(mixedsamples)]
    if len(mixedsamples) - len(cleansamples) < 0:  # Make noise shorter
        cln = cleansamples[:len(mixedsamples)]

    noisesamples = nse
    cleansamples = cln

    return cleansamples, noisesamples, mixedsamples

  except:
    print('error in threads')
    print(mixedpath, cleanpath, noisepath)


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



################################################     Apply N-HANS speech separater     #################################

def apply_demo(cleanpath, noisepath, save_to):

    cleanlist = []
    cleanlist.append(cleanpath)
    noiselist = []
    noiselist.append(noisepath)

    # data processing
    g = tf.Graph()
    with g.as_default():
        with tf.device('/cpu:0'):
            cleanlist = tf.constant(np.array(cleanlist))
            noiselist = tf.constant(np.array(noiselist))

            cleanpath_ph = cleanlist[0]
            noisepath_ph = noiselist[0]

            clean_wav, noise_wav, mix_wav, snr = tf.py_func(combine_signals,
                                                            [cleanpath_ph, noisepath_ph],
                                                            [tf.float32, tf.float32, tf.float32, tf.int32])

            clean_wav, noise_wav, mix_wav = [tf.reshape(x, [-1]) for x in (clean_wav, noise_wav, mix_wav)]

            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

            mix_stft, clean_stft, noise_stft = [tf.contrib.signal.stft(wav, 400, 160, fft_length=400) for wav in [mix_wav, clean_wav, noise_wav]]

            mix_spectrum, clean_spectrum, noise_spectrum = [tf.log(tf.abs(wav_stft) + 1e-5) for wav_stft in [mix_stft, clean_stft, noise_stft]]

            mix_phase = tf.angle(mix_stft)


            # crop data
            mix_spectra = strided_crop(mix_spectrum[Noise_Win:], Mix_Win, 1)

            # postive noise & negative noise
            clean_spectrum = clean_spectrum[:Noise_Win]
            clean_spectrum = tf.reshape(clean_spectrum, [Noise_Win, clean_spectrum.shape[1].value])
            clean_spectra = tf.tile(tf.expand_dims(clean_spectrum, 0), [tf.shape(mix_spectra)[0], 1, 1])

            noise_spectrum = noise_spectrum[:Noise_Win]
            noise_spectrum = tf.reshape(noise_spectrum, [Noise_Win, noise_spectrum.shape[1].value])
            noise_spectra = tf.tile(tf.expand_dims(noise_spectrum, 0), [tf.shape(mix_spectra)[0], 1, 1])

            mixedphs = strided_crop(mix_phase[Noise_Win:], 1, 1)

            # get data
            # print('--------------------------------')
            mix_spectra_, clean_spectra_, noise_spectra_ = sess.run([mix_spectra, clean_spectra, noise_spectra])
            mixedphs_ = sess.run(mixedphs, feed_dict={cleanpath_ph: cleanpath,
                                                      noisepath_ph: noisepath})



    mb = 100
    batches = int(math.ceil(len(mix_spectra_) / float(mb)))
    # batches = 1
    denoised = []
    mix_centers = []

    ## denoiseing
    # graph
    eg = tf.Graph()
    with eg.as_default():
        esess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        with tf.device('/gpu:0'):
            in_ph = [tf.placeholder(dtype=tf.float32, shape=[None, 1, 201], name='cleanph'),
                       tf.placeholder(dtype=tf.float32, shape=[None, Mix_Win, 201], name='mixedph'),
                       tf.placeholder(dtype=tf.float32, shape=[None, 1, 201], name='mixedphaseph'),
                       tf.placeholder(dtype=tf.float32, shape=[None, Noise_Win, 201], name='noisecontextph'),
                       tf.placeholder(dtype=tf.float32, shape=[None, Noise_Win, 201], name='cleancontextph'),
                       tf.placeholder(dtype=tf.int32, shape=[None], name='locationph'),
                       tf.placeholder(dtype=tf.string, shape=[None], name='cleanpathph'),
                       tf.placeholder(dtype=tf.string, shape=[None], name='noisepathph'),
                       tf.placeholder(dtype=tf.int32, shape=[None], name='snrph')]
            _, _, outputs = model(in_ph, False)
        esaver = tf.train.Saver(tf.global_variables())

    # load in trained model
    checkpoints_dir = FLAGS.trained_model_dir
    model_name = FLAGS.trained_model_name
    esaver = tf.train.import_meta_graph(os.path.join(checkpoints_dir, model_name + '.meta'))
    esaver.restore(esess, os.path.join(checkpoints_dir, model_name))

    mixed_tensor = eg.get_tensor_by_name('mixedph:0')
    noise_tensor = eg.get_tensor_by_name('noisecontextph:0')
    clean_tensor = eg.get_tensor_by_name('cleancontextph:0')
    denoised_tensor = eg.get_tensor_by_name('add_72:0')

    # nn processing
    for i in range(batches):
        batch_mix_spectrum, batch_clean_spectrum, batch_noise_spectrum = [spectra[i*mb:(i+1)*mb] for spectra in [mix_spectra_, clean_spectra_, noise_spectra_]]

        batch_denoised_ = esess.run(denoised_tensor, feed_dict={mixed_tensor: batch_mix_spectrum,
                                                                noise_tensor: batch_noise_spectrum,
                                                                clean_tensor: batch_clean_spectrum})

        denoised.append(batch_denoised_)
        mix_center = batch_mix_spectrum[:, 17, :]
        mix_centers.append(mix_center)

    # reconstruction
    denoised = np.concatenate(denoised, axis=0)
    mix_centers = np.concatenate(mix_centers, axis=0)

    recover_samples_from_spectrum(denoised, mixedphs_[:, 0, :], save_to)

    verbose = 0
    if verbose:
        save_to = save_to if len(save_to.split('/')[-1]) == 0 else save_to[:-len(save_to.split('/')[-1])]
        mix_save_to = save_to + 'mixed.wav'
        recover_samples_from_spectrum(mix_centers, mixedphs_[:, 0, :], mix_save_to)


def apply_separator(mixedpath, cleanpath, noisepath, save_to):
    mixedlist = []
    mixedlist.append(mixedpath)
    cleanlist = []
    cleanlist.append(cleanpath)
    noiselist = []
    noiselist.append(noisepath)

    # data processing
    g = tf.Graph()
    with g.as_default():
        with tf.device('/cpu:0'):
            mixedlist = tf.constant(np.array(mixedlist))
            cleanlist = tf.constant(np.array(cleanlist))
            noiselist = tf.constant(np.array(noiselist))

            mixedpath_ph = mixedlist[0]
            cleanpath_ph = cleanlist[0]
            noisepath_ph = noiselist[0]

            clean_wav, noise_wav, mix_wav = tf.py_func(handle_signals,
                                                       [mixedpath_ph, cleanpath_ph, noisepath_ph],
                                                       [tf.float32, tf.float32, tf.float32])

            clean_wav, noise_wav, mix_wav = [tf.reshape(x, [-1]) for x in (clean_wav, noise_wav, mix_wav)]

            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

            mix_stft, clean_stft, noise_stft = [tf.contrib.signal.stft(wav, 400, 160, fft_length=400) for wav in [mix_wav, clean_wav, noise_wav]]

            mix_spectrum, clean_spectrum, noise_spectrum = [tf.log(tf.abs(wav_stft) + 1e-5) for wav_stft in [mix_stft, clean_stft, noise_stft]]

            mix_phase = tf.angle(mix_stft)


            # crop data
            mix_spectra = strided_crop(mix_spectrum, Mix_Win, 1)

            # postive noise & negative noise
            clean_spectrum = clean_spectrum[:Noise_Win]
            clean_spectrum = tf.reshape(clean_spectrum, [Noise_Win, clean_spectrum.shape[1].value])
            clean_spectra = tf.tile(tf.expand_dims(clean_spectrum, 0), [tf.shape(mix_spectra)[0], 1, 1])

            noise_spectrum = noise_spectrum[:Noise_Win]
            noise_spectrum = tf.reshape(noise_spectrum, [Noise_Win, noise_spectrum.shape[1].value])
            noise_spectra = tf.tile(tf.expand_dims(noise_spectrum, 0), [tf.shape(mix_spectra)[0], 1, 1])

            mixedphs = strided_crop(mix_phase, 1, 1)

            # get data
            # print('---------------------------------------------------------------------------------------------')
            mix_spectra_, clean_spectra_, noise_spectra_ = sess.run([mix_spectra, clean_spectra, noise_spectra])
            mixedphs_ = sess.run(mixedphs, feed_dict={mixedpath_ph: mixedpath,
                                                      cleanpath_ph: cleanpath,
                                                      noisepath_ph: noisepath})

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
            in_ph = [tf.placeholder(dtype=tf.float32, shape=[None, 1, 201], name='cleanph'),
                       tf.placeholder(dtype=tf.float32, shape=[None, Mix_Win, 201], name='mixedph'),
                       tf.placeholder(dtype=tf.float32, shape=[None, 1, 201], name='mixedphaseph'),
                       tf.placeholder(dtype=tf.float32, shape=[None, Noise_Win, 201], name='noisecontextph'),
                       tf.placeholder(dtype=tf.float32, shape=[None, Noise_Win, 201], name='cleancontextph'),
                       tf.placeholder(dtype=tf.int32, shape=[None], name='locationph'),
                       tf.placeholder(dtype=tf.string, shape=[None], name='cleanpathph'),
                       tf.placeholder(dtype=tf.string, shape=[None], name='noisepathph'),
                       tf.placeholder(dtype=tf.int32, shape=[None], name='snrph')]
            _, _, outputs = model(in_ph, False)
        esaver = tf.train.Saver(tf.global_variables())

    # load in trained model
    checkpoints_dir = FLAGS.trained_model_dir
    model_name = FLAGS.trained_model_name
    esaver = tf.train.import_meta_graph(os.path.join(checkpoints_dir, model_name + '.meta'))
    esaver.restore(esess, os.path.join(checkpoints_dir, model_name))

    mixed_tensor = eg.get_tensor_by_name('mixedph:0')
    noise_tensor = eg.get_tensor_by_name('noisecontextph:0')
    clean_tensor = eg.get_tensor_by_name('cleancontextph:0')
    denoised_tensor = eg.get_tensor_by_name('add_72:0')

    # nn processing
    for i in range(batches):
        batch_mix_spectrum, batch_clean_spectrum, batch_noise_spectrum = [spectra[i*mb:(i+1)*mb] for spectra in [mix_spectra_, clean_spectra_, noise_spectra_]]

        batch_denoised_ = esess.run(denoised_tensor, feed_dict={mixed_tensor: batch_mix_spectrum,
                                                                noise_tensor: batch_noise_spectrum,
                                                                clean_tensor: batch_clean_spectrum})

        denoised.append(batch_denoised_)
        mix_center = batch_mix_spectrum[:, 17, :]
        mix_centers.append(mix_center)

    # reconstruction
    denoised = np.concatenate(denoised, axis=0)
    mix_centers = np.concatenate(mix_centers, axis=0)

    recover_samples_from_spectrum(denoised, mixedphs_[:, 0, :], save_to)
    verbose = 0
    if verbose:
        save_to = save_to if len(save_to.split('/')[-1]) == 0 else save_to[:-len(save_to.split('/')[-1])]
        mix_save_to = save_to + 'mixed.wav'
        recover_samples_from_spectrum(mix_centers, mixedphs_[:, 0, :], mix_save_to)


if __name__=='__main__':
    mixedpath = FLAGS.input
    noisepath = FLAGS.neg
    cleanpath = FLAGS.pos
    save_to = FLAGS.output

    if os.path.isdir(mixedpath) and os.path.isdir(cleanpath) and os.path.isdir(noisepath) and os.path.isdir(save_to):
        print('##########################################')
        print(' Applying N-HANS source separation system ')
        print('##########################################')

        for file in os.listdir(mixedpath):
            mixedwav = os.path.join(mixedpath, file)
            negwav = os.path.join(noisepath, file)
            poswav = os.path.join(cleanpath, file)
            savewav = os.path.join(save_to, file)
            print('Processing:')
            print('  Input : {}'.format(mixedwav))
            print('  +rec  : {}'.format(poswav))
            print('  -rec  : {}'.format(negwav))
            apply_separator(mixedwav, poswav, negwav, savewav)
            print('Processed result:')
            print('  Output: {}'.format(savewav))
            print('----------------------------------')
            print('')

    elif mixedpath.endswith('.wav') and cleanpath.endswith('.wav') and noisepath.endswith('.wav') and save_to.endswith('.wav'):
        print('##########################################')
        print(' Applying N-HANS source separation system ')
        print('##########################################')
        print('Processing:')
        print('  Input : {}'.format(mixedpath))
        print('  +rec  : {}'.format(cleanpath))
        print('  -rec  : {}'.format(noisepath))
        apply_separator(mixedpath, cleanpath, noisepath, save_to)
        print('Processed result:')
        print('  Output: {}'.format(save_to))
        print('----------------------------------')
        print('')

    else:
        print('Please give the .wav file path or the path to folders containing .wav files')







