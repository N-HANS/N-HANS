########################################################################################################################
#                                          N-HANS speech separator: apply                                              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#   Discription:      Apply N-HANS trained models for audio.                                                           #
#   Authors:          Shuo Liu, Gil Keren, Bjoern Schuller                                                             #
#   Afficiation:      Chair of Embedded Intelligence for Health Care and Wellbeing, University of Augsburg (UAU)  #
#   Date and Time:    May. 04, 2020                                                                                    #
#   Modified:         xxx                                                                                              #
#   Version:          1.5                                                                                              #
#   Dependence Files: xxx                                                                                              #
#   Contact:          shuo.liu@informatik.uni-augburg.de                                                               #
########################################################################################################################

from __future__ import division, absolute_import, print_function
import math, functools
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
if int(tf.__version__.split('.')[0]) == 2:
    import tensorflow.compat.v1 as tf
    tf.compat.v1.disable_v2_behavior()
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from main import model
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_string('input', './audio_examples/mixed.wav', '')
tf.compat.v1.flags.DEFINE_string('neg', './audio_examples/noise_speaker.wav', '')
tf.compat.v1.flags.DEFINE_string('pos', './audio_examples/target_speaker.wav', '')
tf.compat.v1.flags.DEFINE_string('output', './audio_examples/denoised.wav', '')
tf.compat.v1.flags.DEFINE_float('compensate', 0, '')
tf.compat.v1.flags.DEFINE_boolean('ac', False, '')

Noise_Win = 200
Mix_Win = 35


"""
Assistence Functions
"""


def read_wav(in_path):
    rate, samples = wavread(in_path)
    assert rate == FLAGS.Fs
    assert samples.dtype == 'int16'
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
    pnoise = sum(abs(nse) * abs(nse)) / nse.shape[0]

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
        win_samples = int(FLAGS.Fs * 0.025)
        hop_samples = int(FLAGS.Fs * 0.010)
        cleansamples = cleansamples[:-((len(cleansamples) - win_samples) % hop_samples)]

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
        win_samples = int(FLAGS.Fs * 0.025)
        hop_samples = int(FLAGS.Fs * 0.010)
        if (len(mixedsamples) - win_samples) % hop_samples != 0:
            mixedsamples = mixedsamples[:-((len(mixedsamples) - win_samples) % hop_samples)]

        return cleansamples, noisesamples, mixedsamples

    except:
        print('error in threads')
        print(mixedpath, cleanpath, noisepath)


def pad_1D_for_windowing(tensor, length):
    len_before = ((length + 1) // 2) - 1
    len_after = length // 2
    return tf.pad(tensor, [[len_before, len_after], [0, 0]])


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


def recover_samples_from_spectrum(logspectrum_stft, spectrum_phase, save_to):
    abs_spectrum = np.exp(logspectrum_stft)
    spectrum = abs_spectrum * (np.exp(1j * spectrum_phase))
    istft_graph = tf.Graph()
    with istft_graph.as_default():
        num_fea = int(FLAGS.Fs * 0.025) / 2 + 1
        frame_length = int(FLAGS.Fs * 0.025)
        frame_step = int(FLAGS.Fs * 0.010)
        stft_ph = tf.placeholder(tf.complex64, shape=(None, num_fea))
        samples = tf.signal.inverse_stft(stft_ph, frame_length, frame_step, frame_length,
                                                 window_fn=tf.signal.inverse_stft_window_fn(frame_step, forward_window_fn=functools.partial(tf.signal.hann_window, periodic=True)))
        istft_sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        samples_ = istft_sess.run(samples, feed_dict={stft_ph: spectrum})
        wavwrite(save_to, FLAGS.Fs, samples_)


"""
Apply N-HANS speech separater
"""


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

            win_samples = int(FLAGS.Fs * 0.025)
            hop_samples = int(FLAGS.Fs * 0.010)
            mix_stft, clean_stft, noise_stft = [tf.signal.stft(wav, win_samples, hop_samples, fft_length=win_samples) for wav in [mix_wav, clean_wav, noise_wav]]

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
            print('--------------------------------')
            mix_spectra_, clean_spectra_, noise_spectra_ = sess.run([mix_spectra, clean_spectra, noise_spectra])
            mixedphs_ = sess.run(mixedphs, feed_dict={cleanpath_ph: cleanpath,
                                                      noisepath_ph: noisepath})

    mb = 100
    batches = int(math.ceil(len(mix_spectra_) / float(mb)))
    # batches = 1
    denoised = []
    mix_centers = []

    # denoising
    # graph
    eg = tf.Graph()
    with eg.as_default():
        esess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        with tf.device('/gpu:0'):
            num_fea = int(FLAGS.Fs * 0.025) / 2 + 1
            in_ph = [tf.placeholder(dtype=tf.float32, shape=[None, 1, num_fea], name='cleanph'),
                       tf.placeholder(dtype=tf.float32, shape=[None, Mix_Win, num_fea], name='mixedph'),
                       tf.placeholder(dtype=tf.float32, shape=[None, 1, num_fea], name='mixedphaseph'),
                       tf.placeholder(dtype=tf.float32, shape=[None, Noise_Win, num_fea], name='noisecontextph'),
                       tf.placeholder(dtype=tf.float32, shape=[None, Noise_Win, num_fea], name='cleancontextph'),
                       tf.placeholder(dtype=tf.int32, shape=[None], name='locationph'),
                       tf.placeholder(dtype=tf.string, shape=[None], name='cleanpathph'),
                       tf.placeholder(dtype=tf.string, shape=[None], name='noisepathph'),
                       tf.placeholder(dtype=tf.int32, shape=[None], name='snrph')]
            _, _, outputs = model(in_ph, False)
        esaver = tf.train.Saver(tf.global_variables())

    checkpoints_dir = './trained_model'
    esaver = tf.train.import_meta_graph(checkpoints_dir + '/=81457_2-545000.meta')
    esaver.restore(esess, checkpoints_dir + '/=81457_2-545000')

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
    mix_save_to = save_to[:-15] + 'mixed_demo.wav'
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

            win_samples = int(FLAGS.Fs * 0.025)
            hop_samples = int(FLAGS.Fs * 0.010)
            mix_stft, clean_stft, noise_stft = [tf.signal.stft(wav, win_samples, hop_samples, fft_length=win_samples) for wav in [mix_wav, clean_wav, noise_wav]]

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
            print('---------------------------------------------------------------------------------------------')
            mix_spectra_, clean_spectra_, noise_spectra_ = sess.run([mix_spectra, clean_spectra, noise_spectra])
            mixedphs_ = sess.run(mixedphs, feed_dict={mixedpath_ph: mixedpath,
                                                      cleanpath_ph: cleanpath,
                                                      noisepath_ph: noisepath})

    mb = 100
    batches = int(math.ceil(len(mix_spectra_) / float(mb)))
    denoised = []
    mix_centers = []

    # denoiseing
    # graph
    eg = tf.Graph()
    with eg.as_default():
        esess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        with tf.device('/gpu:0'):
            num_fea = int(FLAGS.Fs * 0.025) / 2 + 1
            in_ph = [tf.placeholder(dtype=tf.float32, shape=[None, 1, num_fea], name='cleanph'),
                       tf.placeholder(dtype=tf.float32, shape=[None, Mix_Win, num_fea], name='mixedph'),
                       tf.placeholder(dtype=tf.float32, shape=[None, 1, num_fea], name='mixedphaseph'),
                       tf.placeholder(dtype=tf.float32, shape=[None, Noise_Win, num_fea], name='noisecontextph'),
                       tf.placeholder(dtype=tf.float32, shape=[None, Noise_Win, num_fea], name='cleancontextph'),
                       tf.placeholder(dtype=tf.int32, shape=[None], name='locationph'),
                       tf.placeholder(dtype=tf.string, shape=[None], name='cleanpathph'),
                       tf.placeholder(dtype=tf.string, shape=[None], name='noisepathph'),
                       tf.placeholder(dtype=tf.int32, shape=[None], name='snrph')]
            _, _, outputs = model(in_ph, False)
        esaver = tf.train.Saver(tf.global_variables())

    checkpoints_dir = './trained_model'
    # checkpoints_dir = '/home/user/on_gpu/checkpoints/N_HANS___Source_Separation'
    esaver = tf.train.import_meta_graph(checkpoints_dir + '/81457_2-545000.meta')
    esaver.restore(esess, checkpoints_dir + '/81457_2-545000')

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
    mix_save_to = save_to[:-12] + 'mixed_processed.wav'
    recover_samples_from_spectrum(mix_centers, mixedphs_[:, 0, :], mix_save_to)


if __name__ == '__main__':
    # dir = '/nas/staff/data_work/Sure/example_wav/'
    mixedpath = FLAGS.input
    noisepath = FLAGS.neg
    cleanpath = FLAGS.pos
    save_to = FLAGS.output
    # cleanpath = dir + '00079.wav'
    # noisepath = dir + '00278.wav'

    # save_to = dir + 'output_demo.wav'
    # apply_demo(cleanpath, noisepath, save_to)

    # save_to = dir + 'output.wav'
    # mixedpath = dir + 'mixed_int_ss.wav'
    apply_separator(mixedpath, cleanpath, noisepath, save_to)
