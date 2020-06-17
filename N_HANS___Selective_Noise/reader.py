########################################################################################################################
#                                          N-HANS speech denoiser: reader                                              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#   Discription:      Data reader pipeline                                            .                                #
#   Authors:          Shuo Liu, Gil Keren, Bjoern Schuller                                                             #
#   Afficiation:      Chair of Embedded Intelligence for Health Care and Wellbeing, University of Augsburg (UAU)  #
#   Date and Time:    May. 04, 2020                                                                                    #
#   Modified:         xxx                                                                                              #
#   Version:          1.5                                                                                              #
#   Dependence Files: xxx                                                                                              #
#   Contact:          shuo.liu@informatik.uni-augburg.de                                                               #
########################################################################################################################

import numpy as np
import tensorflow as tf
import os
import random
import time
import hashlib
import pickle as p
from scipy.io.wavfile import read as wavread

"""
context_frames  : number of context frames          (reference signal)
window_frames   : number of frames of noisy signal  (input signal) 
random_slices   : number of slices (each slice contains one input signal and reference signals) 
eval_seeds      : 'valid' or 'test'. In training, evaluation is applied for 'valid' dataset, and in test, for 'test' dataset
wav_dump_folder : the folder to save denoised signals
speech_wav_dir  : the folder contains all speech .wav files
noise_wav_dir   : the folder contains all noise .wav files
"""

FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_integer('context_frames', 200, '')
tf.compat.v1.flags.DEFINE_integer('window_frames', 35, '')
tf.compat.v1.flags.DEFINE_integer('random_slices', 50, '')
tf.compat.v1.flags.DEFINE_integer('Fs', 16000, '')
tf.compat.v1.flags.DEFINE_string('eval_seeds', 'valid', '')
tf.compat.v1.flags.DEFINE_string('wav_dump_folder', './wav_dump/', '')
tf.compat.v1.flags.DEFINE_string('speech_wav_dir', './speech_wav_dir/', '')
tf.compat.v1.flags.DEFINE_string('noise_wav_dir', './noise_wav_dir/', '')

"""
Create Pickle List for Training, Evaluation and Test
"""


# create speech pickle list
def create_speech_seeds():
    """
    Create 3 pickle files: train.pkl, valid.pkl and test.pkl for speech audio files
    :param speech_wav_dir: the directory contains three subfolders, train, valid and test, each includes audio .wav files
    """
    speech_wav_dir = FLAGS.speech_wav_dir
    train_folder_clean = speech_wav_dir + 'train'
    valid_folder_clean = speech_wav_dir + 'valid'
    test_folder_clean = speech_wav_dir + 'test'
    trainseeds = []
    validseeds = []
    testseeds = []
    wavfolders = [train_folder_clean, valid_folder_clean, test_folder_clean]
    seedsfolders = [trainseeds, validseeds, testseeds]

    for ii in range(3):
        wavfolder = wavfolders[ii]
        seedfolder = seedsfolders[ii]
        for root, directories, filenames in os.walk(wavfolder):
            for filename in filenames:
                if filename.endswith('.wav'):
                    seedfolder.append(os.path.join(root, filename))

    train_pkl = speech_wav_dir + 'train.pkl'
    valid_pkl = speech_wav_dir + 'valid.pkl'
    test_pkl = speech_wav_dir + 'test.pkl'
    p.dump(trainseeds, open(train_pkl, "wb"))
    p.dump(validseeds, open(valid_pkl, "wb"))
    p.dump(testseeds, open(test_pkl, "wb"))


# create noise pickle list
def create_noise_seeds():
    """
    Create 3 pickle files: train.pkl, valid.pkl and test.pkl for noise audio files
    :param noise_wav_dir: the directory contains three subfolders, train, valid and test, each includes audio .wav files
    """
    noise_wav_dir = FLAGS.noise_wav_dir
    train_folder_clean = noise_wav_dir + 'train'
    valid_folder_clean = noise_wav_dir + 'valid'
    test_folder_clean = noise_wav_dir + 'test'
    trainseeds = []
    validseeds = []
    testseeds = []
    wavfolders = [train_folder_clean, valid_folder_clean, test_folder_clean]
    seedsfolders = [trainseeds, validseeds, testseeds]

    for ii in range(3):
        wavfolder = wavfolders[ii]
        seedfolder = seedsfolders[ii]
        for root, directories, filenames in os.walk(wavfolder):
            for filename in filenames:
                if filename.endswith('.wav'):
                    seedfolder.append(os.path.join(root, filename))

    train_pkl = noise_wav_dir + 'train.pkl'
    valid_pkl = noise_wav_dir + 'valid.pkl'
    test_pkl = noise_wav_dir + 'test.pkl'
    p.dump(trainseeds, open(train_pkl, "wb"))
    p.dump(validseeds, open(valid_pkl, "wb"))
    p.dump(testseeds, open(test_pkl, "wb"))


""" 
Generating Audio Samples 
combine_signals: generating training, valid and test .wav files
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
    while len(cleansamples) - len(nse_pos) > 0:  # Make noise longer
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
    noise_pos_scaled = K_pos * nse_pos  # Scale the noise
    noise_neg_scaled = K_neg * nse_neg  # Scale the noise
    mixed = sig + noise_pos_scaled + noise_neg_scaled  # Mix
    mixed = mixed / (max(abs(mixed)) + 0.000001)  # Normalize
    target = sig + noise_pos_scaled
    target = target / (max(abs(mixed)) + 0.000001)  # Normalize
    noise_pos_signal = noise_pos_scaled / (max(abs(mixed)) + 0.000001)
    noise_neg_signal = noise_neg_scaled / (max(abs(mixed)) + 0.000001)

    return mixed, target, K_pos, K_neg, noise_pos_signal, noise_neg_signal


def combine_signals(istrain, cleanpath, noisepospath, noisenegpath):
    try:
        # Read Wavs
        cleansamples = read_wav(cleanpath)
        noisepossamples = read_wav(noisepospath)
        noisenegsamples = read_wav(noisenegpath)

        # Normalize
        cleansamples = cleansamples / (max(abs(cleansamples)) + 0.000001)
        noisepossamples = noisepossamples / (max(abs(noisepossamples)) + 0.000001)
        noisenegsamples = noisenegsamples / (max(abs(noisenegsamples)) + 0.000001)
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

        if istrain:
            snr_pos = SNRs[random.randint(0, len(SNRs) - 1)]
            snr_neg = SNRs[random.randint(0, len(SNRs) - 1)]
            mixed, target, Kpos, Kneg, noisepossignal, noisenegsignal = domixing(cleansamples, noisepossamples,
                                                                                 noisenegsamples, snr_pos, snr_neg)
            return target, noisepossignal, noisenegsignal, mixed, np.array(snr_pos, dtype=np.int32), np.array(snr_neg,
                                                                                                              dtype=np.int32)

        else:
            # The noise is the beginning of the file, and mix the rest. The snrs for validation and test data depend on data names, and hence consistent during validation and test
            snrid_pos = int(hashlib.md5(cleanpath).hexdigest()[:8], 16) % len(SNRs)
            snrid_neg = int(hashlib.md5(cleanpath).hexdigest()[:6], 16) % len(SNRs)
            snr_pos = SNRs[snrid_pos]
            snr_neg = SNRs[snrid_neg]
            mixed, target, Kpos, Kneg, noisepossignal, noisenegsignal = domixing(cleansamples, noisepossamples,
                                                                                 noisenegsamples, snr_pos, snr_neg)
            return target, noisepossignal, noisenegsignal, mixed, np.array(snr_pos, dtype=np.int32), np.array(snr_neg,
                                                                                                              dtype=np.int32)
    except:
        print('error in threads')
        print(cleanpath, noisepospath, noisenegpath)


######################################     Global FLAGS and Training FLAGS     #########################################
class read_seeds:
    """
    name              : 'train', 'valid' or 'test'. To generate data reader.
    queuesize         : the size of data queue
    min_after_dequeue : minimum seeds left in queue after dequeue operation
    nthreads          : number of threads processing dequeue
    """

    def __init__(self, name, queuesize, min_after_dequeue, nthreads):
        self.Fs = FLAGS.Fs
        self.frame_length = int(self.Fs * 0.025)
        self.frame_step = int(self.Fs * 0.010)
        self.queuesize = queuesize
        self.min_after_dequeue = min_after_dequeue
        self.window_frames = FLAGS.window_frames
        self.random_slices = FLAGS.random_slices
        self.context_frames = FLAGS.context_frames
        self.eval_stride = 1
        self.istrain = name == 'train'
        self.nthreads = nthreads
        self.name = name

        if self.istrain != True:
            name = FLAGS.eval_seeds

        speech_wav_dir = FLAGS.speech_wav_dir
        noise_wav_dir = FLAGS.noise_wav_dir
        if name == 'train':
            self.seedspaths = [speech_wav_dir + 'train.pkl', noise_wav_dir + 'train.pkl']
        elif name == 'valid':
            self.seedspaths = [speech_wav_dir + 'valid.pkl', noise_wav_dir + 'valid.pkl']
        elif name == 'test':
            self.seedspaths = [speech_wav_dir + 'test.pkl', noise_wav_dir + 'test.pkl']

    def preparations(self):
        """
        Preparation of the data queue.
        """
        self.seeds = []
        for seedspath in self.seedspaths:
            if seedspath.endswith('.pkl'):
                with open(seedspath, 'rb') as f:
                    seedlist = p.load(f)
                    seedlist = tf.constant(np.array(seedlist))
            self.seeds.append(seedlist)

        # Construct seeds queues
        self.seedsqs = []
        for seed in self.seeds:
            num_epochs = None if self.istrain else 1
            seedsq = tf.train.input_producer(seed, capacity=50000, shuffle=self.istrain, num_epochs=num_epochs)
            self.epochcounter = None if self.istrain else tf.compat.v1.local_variables()[-1]
            self.seedsqs.append(seedsq)

        # Get the examples from seeds. It is a list of tensors, with first dimension being the example.
        examples = self.get_examples(self.istrain, self.seedsqs)

        # Create examples queue and enqueue the examples
        dtypes = [x.dtype for x in examples]
        shapes = [x.shape.as_list()[1:] for x in examples]
        if self.istrain:
            exq = tf.queue.RandomShuffleQueue(capacity=self.queuesize, min_after_dequeue=self.min_after_dequeue,
                                              dtypes=dtypes, shapes=shapes)
        else:
            exq = tf.queue.FIFOQueue(capacity=self.queuesize, dtypes=dtypes, shapes=shapes)
        enqop = exq.enqueue_many(examples)

        # The dequeue operation and the mb placeholder
        mb = tf.compat.v1.placeholder(dtype=tf.int32, shape=[], name=self.name + '_mb')
        if self.istrain:
            batch = exq.dequeue_many(mb)
        else:
            batch = exq.dequeue_up_to(mb)

        # Create queue runner
        qr = tf.train.QueueRunner(exq, [enqop] * self.nthreads)
        tf.train.add_queue_runner(qr, tf.compat.v1.GraphKeys.QUEUE_RUNNERS)

        # Set some members
        self.exq = exq
        self.batch = batch
        self.mb = mb

    def get_examples(self, istrain, seedsqs):
        """
        get model inputs from audio samples
        :param istrain: boolean value. If True, the examples are randomly trauncated from audio samples; If False, the examples are trauncated frame by frame from audio samples
        :param seedsqs: seed lists of speech and noise .wav files
        :return:
        """
        clean_seed = seedsqs[0].dequeue()
        noise_pos_seed = seedsqs[1].dequeue()
        noise_neg_seed = seedsqs[1].dequeue()

        target_wav, noise_pos_wav, noise_neg_wav, mix_wav, snr_pos, snr_neg = tf.py_func(combine_signals,
                                                                                         [istrain, clean_seed,
                                                                                          noise_pos_seed,
                                                                                          noise_neg_seed],
                                                                                         [tf.float32, tf.float32,
                                                                                          tf.float32, tf.float32,
                                                                                          tf.int32, tf.int32])

        # flatten
        target_wav, noise_pos_wav, noise_neg_wav, mix_wav = [tf.reshape(x, [-1]) for x in
                                                             (target_wav, noise_pos_wav, noise_neg_wav, mix_wav)]

        # stft
        target_fft = tf.signal.stft(target_wav, self.frame_length, self.frame_step, fft_length=self.frame_length)
        noise_pos_fft = tf.signal.stft(noise_pos_wav, self.frame_length, self.frame_step, fft_length=self.frame_length)
        noise_neg_fft = tf.signal.stft(noise_neg_wav, self.frame_length, self.frame_step, fft_length=self.frame_length)
        mix_fft = tf.signal.stft(mix_wav, self.frame_length, self.frame_step, fft_length=self.frame_length)
        # Magnitude
        target_mag, noise_pos_mag, noise_neg_mag, mix_mag = [tf.abs(x) for x in
                                                             [target_fft, noise_pos_fft, noise_neg_fft, mix_fft]]
        # Phase
        # temp_wav = mix_wav
        mix_phase = tf.math.angle(mix_fft)
        target_phase = tf.math.angle(target_fft)
        pos_phase = tf.math.angle(noise_pos_fft)
        neg_phase = tf.math.angle(noise_neg_fft)

        # log
        target_log, noise_pos_log, noise_neg_log, mix_log = [tf.math.log(x + 1e-5) for x in
                                                             [target_mag, noise_pos_mag, noise_neg_mag, mix_mag]]

        if istrain:
            # padding
            target_log, noise_pos_log, noise_neg_log, mix_log, mix_phase, target_phase, pos_phase, neg_phase = [
                self.pad_1D_for_windowing(x, self.window_frames) for x in
                [target_log, noise_pos_log, noise_neg_log, mix_log, mix_phase, target_phase, pos_phase, neg_phase]]

            # Get n random window-label pairs
            targetslices = []
            targetphslices = []
            mixedslices = []
            mixedphslices = []
            posslices = []
            negslices = []
            posphslices = []
            negphslices = []

            noise_neg_slices = []
            noise_pos_slices = []

            for _ in range(self.random_slices):
                # Crop a window
                targetcropped, mixedcropped, mixedphcropped, targetphcropped, poscropped, posphcropped, negcropped, negphcropped, noiseposcontext, noisenegcontext = \
                    self.synchronized_1D_crop(target_log, noise_pos_log, noise_neg_log, mix_log, mix_phase,
                                              target_phase, pos_phase, neg_phase, self.window_frames,
                                              self.context_frames)

                # Add to list
                for (lst, tnsr) in zip(
                        [targetslices, mixedslices, mixedphslices, targetphslices, posslices, posphslices, negslices,
                         negphslices, noise_pos_slices, noise_neg_slices],
                        [targetcropped, mixedcropped, mixedphcropped, targetphcropped, poscropped, posphcropped,
                         negcropped, negphcropped, noiseposcontext, noisenegcontext]):
                    lst.append(tnsr)

            # Stack and return
            return tf.stack(targetslices, axis=0), tf.stack(mixedslices, axis=0), tf.stack(mixedphslices,
                                                                                           axis=0), tf.stack(
                targetphslices, axis=0), tf.stack(posslices, axis=0), tf.stack(posphslices, axis=0), tf.stack(negslices,
                                                                                                              axis=0), tf.stack(
                negphslices, axis=0), \
                   tf.stack(noise_pos_slices, axis=0), tf.stack(noise_neg_slices, axis=0), \
                   tf.constant([0] * self.random_slices, dtype=tf.int32), tf.constant([''] * self.random_slices,
                                                                                      dtype=tf.string), \
                   tf.constant([''] * self.random_slices, dtype=tf.string), tf.constant([''] * self.random_slices,
                                                                                        dtype=tf.string), tf.constant(
                [0] * self.random_slices, dtype=tf.int32), tf.constant([0] * self.random_slices, dtype=tf.int32)
        else:
            targetslices = self.strided_crop(target_log[self.context_frames:], 1, self.eval_stride)
            mixedslices = self.strided_crop(mix_log[self.context_frames:], self.window_frames, self.eval_stride)
            mixedphslices = self.strided_crop(mix_phase[self.context_frames:], 1, self.eval_stride)
            targetphslices = self.strided_crop(target_phase[self.context_frames:], 1, self.eval_stride)
            posslices = self.strided_crop(noise_pos_log[self.context_frames:], self.window_frames, self.eval_stride)
            posphslices = self.strided_crop(pos_phase[self.context_frames:], 1, self.eval_stride)
            negslices = self.strided_crop(noise_neg_log[self.context_frames:], self.window_frames, self.eval_stride)
            negphslices = self.strided_crop(neg_phase[self.context_frames:], 1, self.eval_stride)

            noiseposcontext = noise_pos_log[:self.context_frames]
            noiseposcontext = tf.reshape(noiseposcontext, [self.context_frames, noiseposcontext.shape[1].value])
            noise_pos_slices = tf.tile(tf.expand_dims(noiseposcontext, 0), [tf.shape(targetslices)[0], 1, 1])
            noisenegcontext = noise_neg_log[:self.context_frames]
            noisenegcontext = tf.reshape(noisenegcontext, [self.context_frames, noisenegcontext.shape[1].value])
            noise_neg_slices = tf.tile(tf.expand_dims(noisenegcontext, 0), [tf.shape(targetslices)[0], 1, 1])
            locations = tf.range(0, tf.shape(targetslices)[0], dtype=tf.int32)
            clean_seeds = tf.tile([clean_seed], [tf.shape(targetslices)[0]])
            noise_pos_seeds = tf.tile([noise_pos_seed], [tf.shape(targetslices)[0]])
            noise_neg_seeds = tf.tile([noise_neg_seed], [tf.shape(targetslices)[0]])
            snrs_pos = tf.tile([snr_pos], [tf.shape(targetslices)[0]])
            snrs_neg = tf.tile([snr_neg], [tf.shape(targetslices)[0]])
            return targetslices, mixedslices, mixedphslices, targetphslices, posslices, posphslices, negslices, negphslices, noise_pos_slices, noise_neg_slices, locations, clean_seeds, noise_pos_seeds, noise_neg_seeds, snrs_pos, snrs_neg  # , temp_wav

    def pad_1D_for_windowing(self, tensor, length):
        """
        Padding the input spectrogramm
        :param tensor: shape of [Times x Frequencies]
        :param length: equals to window frames
        :return: padded spectrogram
        """
        len_before = ((length + 1) // 2) - 1
        len_after = length // 2
        return tf.pad(tensor, [[len_before, len_after], [0, 0]])

    def synchronized_1D_crop(self, target, noise_pos, noise_neg, mixed, mixedph, targetph, posph, negph, winlength,
                             contextlength):
        winmaxval = tf.shape(mixed)[0] - winlength
        winstart = tf.random.uniform([], dtype=tf.int32, minval=0, maxval=winmaxval + 1)

        # Crop window
        mixedcropped = mixed[winstart:winstart + winlength]
        targetcropped = target[winstart:winstart + winlength]
        mixedphcropped = mixedph[winstart:winstart + winlength]
        mixedcropped = tf.reshape(mixedcropped, [winlength, mixedcropped.shape[1].value])
        targetcropped = tf.reshape(targetcropped, [winlength, targetcropped.shape[1].value])
        mixedphcropped = tf.reshape(mixedphcropped, [winlength, mixedphcropped.shape[1].value])

        # Get the center of the clean and mixed phase
        targetcropped = targetcropped[self.window_frames // 2]
        targetcropped = tf.expand_dims(targetcropped, 0)
        mixedphcropped = mixedphcropped[self.window_frames // 2]
        mixedphcropped = tf.expand_dims(mixedphcropped, 0)

        # log.....
        poscropped = noise_pos[winstart:winstart + winlength]
        poscropped = tf.reshape(poscropped, [winlength, poscropped.shape[1].value])
        negcropped = noise_neg[winstart:winstart + winlength]
        negcropped = tf.reshape(negcropped, [winlength, negcropped.shape[1].value])

        # ph.....
        targetphcropped = targetph[winstart:winstart + winlength]
        targetphcropped = tf.reshape(targetphcropped, [winlength, targetphcropped.shape[1].value])
        targetphcropped = targetphcropped[self.window_frames // 2]
        targetphcropped = tf.expand_dims(targetphcropped, 0)

        negphcropped = negph[winstart:winstart + winlength]
        negphcropped = tf.reshape(negphcropped, [winlength, negphcropped.shape[1].value])
        negphcropped = negphcropped[self.window_frames // 2]
        negphcropped = tf.expand_dims(negphcropped, 0)

        posphcropped = posph[winstart:winstart + winlength]
        posphcropped = tf.reshape(posphcropped, [winlength, posphcropped.shape[1].value])
        posphcropped = posphcropped[self.window_frames // 2]
        posphcropped = tf.expand_dims(posphcropped, 0)

        # The rest of the noise
        noiseposrest = tf.concat([noise_pos[:winstart], noise_pos[winstart + winlength:]], axis=0)
        noisenegrest = tf.concat([noise_neg[:winstart], noise_neg[winstart + winlength:]], axis=0)

        # Get the noise part
        posrestmaxval = tf.shape(noiseposrest)[0] - contextlength
        posreststart = tf.random.uniform([], dtype=tf.int32, minval=0, maxval=posrestmaxval + 1)
        noiseposcontext = noiseposrest[posreststart:posreststart + contextlength]
        noiseposcontext = tf.reshape(noiseposcontext, [contextlength, noiseposcontext.shape[1].value])

        negrestmaxval = tf.shape(noisenegrest)[0] - contextlength
        negreststart = tf.random.uniform([], dtype=tf.int32, minval=0, maxval=negrestmaxval + 1)
        noisenegcontext = noisenegrest[negreststart:negreststart + contextlength]
        noisenegcontext = tf.reshape(noisenegcontext, [contextlength, noisenegcontext.shape[1].value])

        return targetcropped, mixedcropped, mixedphcropped, targetphcropped, poscropped, posphcropped, negcropped, negphcropped, noiseposcontext, noisenegcontext

    def strided_crop(self, tensor, length, stride):
        # we assume that we have a length dimension and a feature dimension
        assert len(tensor.shape) == 2
        n_features = int(tensor.shape[1])
        padded = self.pad_1D_for_windowing(tensor, length)
        windows = tf.extract_image_patches(tf.expand_dims(tf.expand_dims(padded, axis=0), axis=3),
                                           ksizes=[1, length, n_features, 1],
                                           strides=[1, stride, n_features, 1],
                                           rates=[1, 1, 1, 1],
                                           padding='VALID')
        return tf.reshape(windows, [-1, length, n_features])

    def get_inputs(self):
        if not hasattr(self, 'ph'):
            num_features = int(self.frame_length / 2) + 1
            self.ph = [tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 1, num_features], name='targetph'),
                       tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.window_frames, num_features],
                                                name='mixedph'),
                       tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 1, num_features], name='mixedphaseph'),
                       tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 1, num_features], name='targetphaseph'),
                       tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.window_frames, num_features], name='posph'),
                       tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 1, num_features], name='posphaseph'),
                       tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.window_frames, num_features], name='negph'),
                       tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 1, num_features], name='negphaseph'),
                       tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.context_frames, num_features],
                                                name='noiseposcontextph'),
                       tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.context_frames, num_features],
                                                name='noisenegcontextph'),
                       tf.compat.v1.placeholder(dtype=tf.int32, shape=[None], name='locationph'),
                       tf.compat.v1.placeholder(dtype=tf.string, shape=[None], name='targetpathph'),
                       tf.compat.v1.placeholder(dtype=tf.string, shape=[None], name='noisepospathph'),
                       tf.compat.v1.placeholder(dtype=tf.string, shape=[None], name='noisenegpathph'),
                       tf.compat.v1.placeholder(dtype=tf.int32, shape=[None], name='snrposph'),
                       tf.compat.v1.placeholder(dtype=tf.int32, shape=[None], name='snrnegph'),
                       ]
        return self.ph


if __name__ == '__main__':
    g = tf.Graph()
    with g.as_default():
        reader = read_seeds(name='valid', queuesize=200, min_after_dequeue=100, nthreads=16)
        with tf.device('/cpu:0'):
            reader.preparations()

        print('starting queue runners')
        sess = tf.compat.v1.Session(graph=g, config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
        sess.run(tf.compat.v1.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        queue_stats = tf.stack(
            [x.queue.size() for x in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.QUEUE_RUNNERS)])
        feed_dict = {reader.mb: 100}

        with tf.device('/gpu:0'):
            target, mixed, mixedph, targetph, pos, posph, neg, negph, noiseposcontext, noisenegcontext, location, cleanpath, noisepospath, noisenegpath, snr_pos, snr_neg = reader.get_inputs()

        for i in range(3):
            print(i)
            t1 = time.time()
            target_, mixed_, mixedph_, targetph_, pos_, posph_, neg_, negph_, noiseposcontext_, noisenegcontext_, location_, cleanpath_, noisepospath_, noisenegpath_, snr_pos_, snr_neg_ = sess.run(
                reader.batch, feed_dict=feed_dict)
            t2 = time.time()
            print(sess.run(queue_stats), t2 - t1)
