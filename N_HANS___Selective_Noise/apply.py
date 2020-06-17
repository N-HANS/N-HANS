########################################################################################################################
#                                          N-HANS speech denoiser: apply                                               #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#   Discription:      Apply N-HANS trained models for audio.                          .                                #
#   Authors:          Shuo Liu, Gil Keren, Bjoern Schuller                                                             #
#   Afficiation:      ZD.B Chair of Embedded Intelligence for Health Care and Wellbeing, University of Augsburg (UAU)  #
#   Date and Time:    May. 04, 2020                                                                                    #
#   Modified:         xxx                                                                                              #
#   Version:          1.5                                                                                              #
#   Dependence Files: xxx                                                                                              #
#   Contact:          shuo.liu@informatik.uni-augburg.de                                                               #
########################################################################################################################


from __future__ import division, absolute_import, print_function
import numpy as np, functools
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
if int(tf.__version__.split('.')[0]) == 2:
    import tensorflow.compat.v1 as tf
    tf.compat.v1.disable_v2_behavior()
import math
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from main import model


FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_string('input', './audio_examples/mixed.wav', '')
tf.compat.v1.flags.DEFINE_string('neg', './audio_examples/game_noise.wav', '')
tf.compat.v1.flags.DEFINE_string('pos', './audio_examples/Silent.wav', '')
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
        win_samples = int(FLAGS.Fs * 0.025)
        hop_samples = int(FLAGS.Fs * 0.010)
        if (len(cleansamples) - win_samples) % hop_samples != 0:
            cleansamples = cleansamples[:-((len(cleansamples) - win_samples) % hop_samples)]

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
        win_samples = int(FLAGS.Fs * 0.025)
        hop_samples = int(FLAGS.Fs * 0.010)
        if (len(mixedsamples) - win_samples) % hop_samples != 0:
            mixedsamples = mixedsamples[:-((len(mixedsamples) - win_samples) % hop_samples)]

        return noisepossamples, noisenegsamples, mixedsamples

    except:
        print('error in threads')
        print(mixedpath, noisepospath, noisenegpath)


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
        samples = tf.signal.inverse_stft(stft_ph, frame_length, frame_step, frame_length, window_fn=tf.signal.inverse_stft_window_fn(frame_step, forward_window_fn=functools.partial(tf.signal.hann_window, periodic=True)))
        istft_sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        samples_ = istft_sess.run(samples, feed_dict={stft_ph: spectrum})
        wavwrite(save_to, FLAGS.Fs, samples_)

    return samples_


"""
Apply N-HANS speech denoiser
"""


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

            win_samples = int(FLAGS.Fs * 0.025)
            hop_samples = int(FLAGS.Fs * 0.010)
            mix_stft, pos_stft, neg_stft = [tf.signal.stft(wav, win_samples, hop_samples, fft_length=win_samples) for wav in [mix_wav, noise_pos_wav, noise_neg_wav]]

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
            print('--------------------------------')
            mix_spectra_, pos_spectra_, neg_spectra_ = sess.run([mix_spectra, pos_spectra, neg_spectra])
            mixedphs_ = sess.run(mixedphs, feed_dict={speechpath_ph: speechpath,
                                                      pospath_ph: pospath,
                                                      negpath_ph: negpath})


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
            in_ph = [tf.placeholder(dtype=tf.float32, shape=[None, 1, num_fea], name='targetph'),
                  tf.placeholder(dtype=tf.float32, shape=[None, Mix_Win, num_fea], name='mixedph'),
                  tf.placeholder(dtype=tf.float32, shape=[None, 1, num_fea], name='mixedphaseph'),
                  tf.placeholder(dtype=tf.float32, shape=[None, 1, num_fea], name='targetphaseph'),
                  tf.placeholder(dtype=tf.float32, shape=[None, Mix_Win, num_fea], name='posph'),
                  tf.placeholder(dtype=tf.float32, shape=[None, 1, num_fea], name='posphaseph'),
                  tf.placeholder(dtype=tf.float32, shape=[None, Mix_Win, num_fea], name='negph'),
                  tf.placeholder(dtype=tf.float32, shape=[None, 1, num_fea], name='negphaseph'),
                  tf.placeholder(dtype=tf.float32, shape=[None, Noise_Win, num_fea], name='noiseposcontextph'),
                  tf.placeholder(dtype=tf.float32, shape=[None, Noise_Win, num_fea], name='noisenegcontextph'),
                  tf.placeholder(dtype=tf.int32, shape=[None], name='locationph'),
                  tf.placeholder(dtype=tf.string, shape=[None], name='targetpathph'),
                  tf.placeholder(dtype=tf.string, shape=[None], name='noisepospathph'),
                  tf.placeholder(dtype=tf.string, shape=[None], name='noisenegpathph'),
                  tf.placeholder(dtype=tf.int32, shape=[None], name='snrposph'),
                  tf.placeholder(dtype=tf.int32, shape=[None], name='snrnegph'),
                  ]

            _, _, outputs = model(in_ph, False)
        esaver = tf.train.Saver(tf.global_variables())

    checkpoints_dir = './trained_model'
    esaver = tf.train.import_meta_graph(checkpoints_dir + '/81448_0-1000000.meta')
    esaver.restore(esess, checkpoints_dir + '/81448_0-1000000')

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

            win_samples = int(FLAGS.Fs * 0.025)
            hop_samples = int(FLAGS.Fs * 0.010)
            mix_stft, pos_stft, neg_stft = [tf.signal.stft(wav, win_samples, hop_samples, fft_length=win_samples) for wav in
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
            print('------------------------------------------------------------------------------------')
            mix_spectra_, pos_spectra_, neg_spectra_ = sess.run([mix_spectra, pos_spectra, neg_spectra])
            mixedphs_ = sess.run(mixedphs, feed_dict={mixedpath_ph: mixedpath,
                                                      pospath_ph: pospath,
                                                      negpath_ph: negpath})

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
            in_ph = [tf.placeholder(dtype=tf.float32, shape=[None, 1, num_fea], name='targetph'),
                     tf.placeholder(dtype=tf.float32, shape=[None, Mix_Win, num_fea], name='mixedph'),
                     tf.placeholder(dtype=tf.float32, shape=[None, 1, num_fea], name='mixedphaseph'),
                     tf.placeholder(dtype=tf.float32, shape=[None, 1, num_fea], name='targetphaseph'),
                     tf.placeholder(dtype=tf.float32, shape=[None, Mix_Win, num_fea], name='posph'),
                     tf.placeholder(dtype=tf.float32, shape=[None, 1, num_fea], name='posphaseph'),
                     tf.placeholder(dtype=tf.float32, shape=[None, Mix_Win, num_fea], name='negph'),
                     tf.placeholder(dtype=tf.float32, shape=[None, 1, num_fea], name='negphaseph'),
                     tf.placeholder(dtype=tf.float32, shape=[None, Noise_Win, num_fea], name='noiseposcontextph'),
                     tf.placeholder(dtype=tf.float32, shape=[None, Noise_Win, num_fea], name='noisenegcontextph'),
                     tf.placeholder(dtype=tf.int32, shape=[None], name='locationph'),
                     tf.placeholder(dtype=tf.string, shape=[None], name='targetpathph'),
                     tf.placeholder(dtype=tf.string, shape=[None], name='noisepospathph'),
                     tf.placeholder(dtype=tf.string, shape=[None], name='noisenegpathph'),
                     tf.placeholder(dtype=tf.int32, shape=[None], name='snrposph'),
                     tf.placeholder(dtype=tf.int32, shape=[None], name='snrnegph'),
                     ]
            _, _, outputs = model(in_ph, False)
        esaver = tf.train.Saver(tf.global_variables())

    checkpoints_dir = './trained_model'
    esaver = tf.train.import_meta_graph(checkpoints_dir + '/81448_0-1000000.meta')
    esaver.restore(esess, checkpoints_dir + '/81448_0-1000000')

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
    mix_save_to = save_to[:-12] + 'mixed_processed.wav'
    mixed_samples = recover_samples_from_spectrum(mix_centers, mixedphs_[:, 0, :], mix_save_to)
    removed_save_to = save_to[:-12] + 'removed.wav'
    removed_samples = mixed_samples - denoised_samples
    wavwrite(removed_save_to, FLAGS.Fs, removed_samples)

    snr_est = np.mean(np.square(denoised_samples)) / np.mean(np.square(removed_samples))
    print(snr_est)
    print('---------------------------')
    if not FLAGS.ac:
        compensation_factor = FLAGS.compensate
    else:
        compensation_factor = snr_est/20
    compensated_save_to = save_to[:-12] + 'compensated.wav'
    compensated_samples = denoised_samples + removed_samples * compensation_factor
    wavwrite(compensated_save_to, FLAGS.Fs, compensated_samples)

    # snr = S/N   snr_est = S-e / N+e
    # e is distortion, need to be compensate. For a definite snr, Lower snr_est requires more compensation.


def apply_denoiser(mixedpath, negpath, save_to):
    dir = './audio_examples/'
    pospath = dir + 'Silent.wav'
    apply_snc(mixedpath, pospath, negpath, save_to)


# def main():
#     mixedpath = FLAGS.input
#     negpath = FLAGS.neg
#     save_to = FLAGS.output
#
#     apply_denoiser(mixedpath, negpath, save_to)

if __name__ == '__main__':

    # dir = '/nas/staff/data_work/Sure/example_wav/'
    # demo = 3
    # if demo == 1:
    #     speechpath = dir + '6930-76324-0008.wav'
    #     pospath = dir + '5Qv2VEX9iyI_29.000.wav'
    #     negpath = dir + 'A67lyBRPevM_0.000.wav'
    #
    # elif demo == 2:
    #     speechpath = dir + '121-121726-0005.wav'
    #     pospath = dir + 'Imx5o81QWk0_120.000.wav'
    #     negpath = dir + '3za2WvNjiBk_0.000.wav'
    #
    # elif demo == 3:
    #     speechpath = dir + '121-121726-0005.wav'
    #     negpath = dir + '3za2WvNjiBk_0.000.wav'
    #
    #
    # # save_to = dir + 'output_demo.wav'
    # # apply_demo(speechpath, pospath, negpath, save_to)
    #
    # save_to = dir + 'output.wav'
    # mixedpath = dir + 'mixed_int.wav'
    # # apply_snc(mixedpath, pospath, negpath, save_to)
    #
    #
    # mixedpath = dir + 'zhaolei_section.wav'
    # fs, samples = wavread(mixedpath)
    # negpath = dir + 'zhaolei_baseline.wav'
    # save_to = dir + 'zhaolei.wav'
    #
    mixedpath = FLAGS.input
    negpath = FLAGS.neg
    save_to = FLAGS.output

    apply_denoiser(mixedpath, negpath, save_to)
