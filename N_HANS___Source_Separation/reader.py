########################################################################################################################
#                                          N-HANS speech separator: reader                                             #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#   Description:      Data reader pipeline.                                                                            #
#   Authors:          Shuo Liu, Gil Keren, Bjoern Schuller                                                             #
#   Affiliation:      ZD.B Chair of Embedded Intelligence for Health Care and Wellbeing, University of Augsburg (UAU)  #
#   Version:          1.0                                                                                              #
#   Last Update:      Nov. 14, 2019                                                                                    #
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

#################################################     FLAGS     ########################################################
'''
context_frames  : number of context frames          (reference signal)
window_frames   : number of frames of noisy signal  (input signal) 
random_slices   : number of slices (each slice contains one input signal and reference signals) 
eval_seeds      : 'valid' or 'test'. In training, evaluation is applied for 'valid' dataset, and in test, for 'test' dataset
wav_dump_folder : the folder to save denoised signals
'''
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('context_frames', 200, '')
tf.app.flags.DEFINE_integer('window_frames', 35, '')
tf.app.flags.DEFINE_integer('random_slices', 50, '')
tf.app.flags.DEFINE_string('eval_seeds', 'valid', '')
tf.app.flags.DEFINE_string('wav_dump_folder', './wav_dump/', '')
tf.app.flags.DEFINE_string('speech_wav_dir', './speech_wav_dir/', '')


################################     Create Pickle List for Training, Evaluation and Test     ##########################
def create_seeds():
    '''
    Create 3 pickle files: train.pkl, valid.pkl and test.pkl for speech audio files
    :param speech_wav_dir: the directory contains three subfolders, train, valid and test, each includes audio .wav files
    '''
    speech_wav_dir=FLAGS.speech_wav_dir
    train_folder_clean = os.path.join(speech_wav_dir, 'train')
    valid_folder_clean = os.path.join(speech_wav_dir, 'valid')
    test_folder_clean = os.path.join(speech_wav_dir, 'test')
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

    train_pkl = os.path.join(speech_wav_dir, 'train.pkl')
    valid_pkl = os.path.join(speech_wav_dir, 'valid.pkl')
    test_pkl = os.path.join(speech_wav_dir, 'test.pkl')
    p.dump(trainseeds, open(train_pkl, "wb"))
    p.dump(validseeds, open(valid_pkl, "wb"))
    p.dump(testseeds, open(test_pkl, "wb"))

#################################     Additional Functions for Generating Audio Samples     ############################
'''
combine_signals: generating training, valid and test .wav files
'''
def read_wav(in_path):
  rate, samples = wavread(in_path)
  assert rate == 16000
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


def combine_signals(istrain, cleanpath, noisepath):
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

    if istrain:
      snr = SNRs[random.randint(0, len(SNRs) - 1)]
      mixed, K = domixing(cleansamples, noisesamples, snr)
      return cleansamples, noisesamples * K, mixed, np.array(snr, dtype=np.int32)

    else:
      # The noise is the beginning of the file, and mix the rest
      snrid = int(hashlib.md5(cleanpath).hexdigest()[:8], 16) % len(SNRs)
      snr = SNRs[snrid]
      mixed, K = domixing(cleansamples, noisesamples, snr)
      return cleansamples, noisesamples * K, mixed, np.array(snr, dtype=np.int32)

  except:
    print('error in threads')
    print(cleanpath, noisepath)


######################################     Global FLAGS and Training FLAGS     #########################################
class read_seeds:
    '''
    name              : 'train', 'valid' or 'test'. To generate data reader.
    queuesize         : the size of data queue
    min_after_dequeue : minimum seeds left in queue after dequeue operation
    nthreads          : number of threads processing dequeue
    '''
    def __init__(self, name, queuesize, min_after_dequeue, nthreads):
        self.Fs = 16000
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
        if name == 'train':
            self.seedspath = os.path.join(speech_wav_dir, 'train.pkl')
        elif name == 'valid':
            self.seedspath = os.path.join(speech_wav_dir, 'valid.pkl')
        elif name == 'test':
            self.seedspath = os.path.join(speech_wav_dir, 'test.pkl')


    def preparations(self):
        '''
        Preparation of the data queue.
        '''
        self.seeds = []
        with open(self.seedspath, 'rb') as f:
            seedlist = p.load(f)
            seedlist = tf.constant(np.array(seedlist))
            self.seeds.append(seedlist)

        # Construct seeds queues
        self.seedsqs = []
        for i, seed in enumerate(self.seeds):
            num_epochs = None if self.istrain else 1
            seedsq = tf.train.input_producer(seed, capacity=50000, shuffle=self.istrain, num_epochs=num_epochs)
            self.epochcounter = None if self.istrain else tf.local_variables()[-1]
            self.seedsqs.append(seedsq)

        # Get the examples from seeds. It is a list of tensors, with first dimension being the example.
        examples = self.get_examples(self.istrain, self.seedsqs)

        # Create examples queue and enqueue the examples
        dtypes = [x.dtype for x in examples]
        shapes = [x.shape.as_list()[1:] for x in examples]
        if self.istrain:
            exq = tf.RandomShuffleQueue(capacity=self.queuesize, min_after_dequeue=self.min_after_dequeue,
                                      dtypes=dtypes, shapes=shapes)
        else:
            exq = tf.FIFOQueue(capacity=self.queuesize, dtypes=dtypes, shapes=shapes)
        enqop = exq.enqueue_many(examples)

        # The dequeue operation and the mb placeholder
        mb = tf.placeholder(dtype=tf.int32, shape=[], name=self.name + '_mb')
        if self.istrain:
            batch = exq.dequeue_many(mb)
        else:
            batch = exq.dequeue_up_to(mb)

        # Create queue runner
        qr = tf.train.QueueRunner(exq, [enqop] * self.nthreads)
        tf.train.add_queue_runner(qr, tf.GraphKeys.QUEUE_RUNNERS)

        # Set some members
        self.exq = exq
        self.batch = batch
        self.mb = mb


    def get_examples(self, istrain, seedsqs):
        '''
        get model inputs from audio samples
        :param istrain: boolean value. If True, the examples are randomly trauncated from audio samples; If False, the examples are trauncated frame by frame from audio samples
        :param seedsqs: seed list of speech files
        :return:
        '''
        target_seed = seedsqs[0].dequeue()
        noise_seed = seedsqs[0].dequeue()
        target_wav, noise_wav, mix_wav, snr = tf.py_func(combine_signals, [istrain, target_seed, noise_seed], [tf.float32, tf.float32, tf.float32, tf.int32])

        # flatten
        target_wav, noise_wav, mix_wav = [tf.reshape(x, [-1]) for x in (target_wav, noise_wav, mix_wav)]

        # stft
        target_fft = tf.contrib.signal.stft(target_wav, self.frame_length, self.frame_step, fft_length=self.frame_length)
        noise_fft = tf.contrib.signal.stft(noise_wav, self.frame_length, self.frame_step, fft_length=self.frame_length)
        mix_fft = tf.contrib.signal.stft(mix_wav, self.frame_length, self.frame_step, fft_length=self.frame_length)

        # Magnitude
        target_mag, noise_mag, mix_mag = [tf.abs(x) for x in [target_fft, noise_fft, mix_fft]]
        # Phase
        mix_phase = tf.angle(mix_fft)
        # log
        target_log, noise_log, mix_log = [tf.log(x + 1e-5) for x in [target_mag, noise_mag, mix_mag]]

        if istrain:
            # padding
            target_log, noise_log, mix_log, mix_phase = [self.pad_1D_for_windowing(x, self.window_frames) for x in [target_log, noise_log, mix_log, mix_phase]]

            # Get n random window-label pairs
            targetslices = []
            mixedslices = []
            mixedphslices = []
            noise_context_slices = []
            clean_context_slices = []
            for _ in range(self.random_slices):
                # Crop a window
                cleancropped, mixedcropped, mixedphcropped, noisecontext, cleancontext = \
                    self.synchronized_1D_crop(target_log, noise_log, mix_log, mix_phase, self.window_frames, self.context_frames)

                # Add to list
                for (lst, tnsr) in zip([targetslices, mixedslices, mixedphslices, noise_context_slices, clean_context_slices], [cleancropped, mixedcropped, mixedphcropped, noisecontext, cleancontext]):
                    lst.append(tnsr)

            # Stack and return
            return tf.stack(targetslices, axis=0), tf.stack(mixedslices, axis=0), tf.stack(mixedphslices, axis=0), \
                   tf.stack(noise_context_slices, axis=0), tf.stack(clean_context_slices, axis=0), \
                tf.constant([0] * self.random_slices, dtype=tf.int32), tf.constant([''] * self.random_slices, dtype=tf.string), \
                tf.constant([''] * self.random_slices, dtype=tf.string), tf.constant([0] * self.random_slices, dtype=tf.int32)
        else:
            targetslices = self.strided_crop(target_log[self.context_frames:], 1, self.eval_stride)
            mixedslices = self.strided_crop(mix_log[self.context_frames:], self.window_frames, self.eval_stride)
            mixedphslices = self.strided_crop(mix_phase[self.context_frames:], 1, self.eval_stride)
            noisecontext = noise_log[:self.context_frames]
            noisecontext = tf.reshape(noisecontext, [self.context_frames, noisecontext.shape[1].value])
            noise_context_slices = tf.tile(tf.expand_dims(noisecontext, 0), [tf.shape(targetslices)[0], 1, 1])
            cleancontext = target_log[:self.context_frames]
            cleancontext= tf.reshape(cleancontext, [self.context_frames, cleancontext.shape[1].value])
            clean_context_slices = tf.tile(tf.expand_dims(cleancontext, 0), [tf.shape(targetslices)[0], 1, 1])
            locations = tf.range(0, tf.shape(targetslices)[0], dtype=tf.int32)
            target_seeds = tf.tile([target_seed], [tf.shape(targetslices)[0]])
            noise_seeds = tf.tile([noise_seed], [tf.shape(targetslices)[0]])
            snrs = tf.tile([snr], [tf.shape(targetslices)[0]])
            return targetslices, mixedslices, mixedphslices, noise_context_slices, clean_context_slices, locations, target_seeds, noise_seeds, snrs


    def pad_1D_for_windowing(self, tensor, length):
        '''
        Padding the input spectrogramm
        :param tensor: shape of [Times x Frequencies]
        :param length: equals to window frames
        :return: padded spectrogram
        '''
        len_before = ((length + 1) // 2) - 1
        len_after = length // 2
        return tf.pad(tensor, [[len_before, len_after],[0,0]])


    def synchronized_1D_crop(self, clean, noise, mixed, mixedph, winlength, contextlength):
        winmaxval = tf.shape(mixed)[0] - winlength
        winstart = tf.random_uniform([], dtype=tf.int32, minval=0, maxval=winmaxval+1)

        # Crop window
        mixedcropped = mixed[winstart:winstart+winlength]
        cleancropped = clean[winstart:winstart+winlength]
        mixedphcropped = mixedph[winstart:winstart+winlength]
        mixedcropped = tf.reshape(mixedcropped, [winlength, mixedcropped.shape[1].value])
        cleancropped = tf.reshape(cleancropped, [winlength, cleancropped.shape[1].value])
        mixedphcropped = tf.reshape(mixedphcropped, [winlength, mixedphcropped.shape[1].value])

        # Get the center of the clean and mixed phase
        cleancropped = cleancropped[self.window_frames // 2]
        cleancropped = tf.expand_dims(cleancropped, 0)
        mixedphcropped = mixedphcropped[self.window_frames // 2]
        mixedphcropped = tf.expand_dims(mixedphcropped, 0)

        # The rest of the noise
        cleanrest = tf.concat([clean[:winstart], clean[winstart+winlength:]], axis=0)
        noiserest = tf.concat([noise[:winstart], noise[winstart+winlength:]], axis=0)

        # Get the noise part
        restmaxval = tf.shape(noiserest)[0] - contextlength
        reststart = tf.random_uniform([], dtype=tf.int32, minval=0, maxval=restmaxval+1)
        noisecontext = noiserest[reststart:reststart + contextlength]
        noisecontext = tf.reshape(noisecontext, [contextlength, noisecontext.shape[1].value])

        restmaxval = tf.shape(cleanrest)[0] - contextlength
        reststart = tf.random_uniform([], dtype=tf.int32, minval=0, maxval=restmaxval+1)
        cleancontext = cleanrest[reststart:reststart + contextlength]
        cleancontext = tf.reshape(cleancontext, [contextlength, cleancontext.shape[1].value])

        return cleancropped, mixedcropped, mixedphcropped, noisecontext, cleancontext


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
            self.ph = [tf.placeholder(dtype=tf.float32, shape=[None, 1, 201], name='cleanph'),
                       tf.placeholder(dtype=tf.float32, shape=[None, self.window_frames, 201], name='mixedph'),
                       tf.placeholder(dtype=tf.float32, shape=[None, 1, 201], name='mixedphaseph'),
                       tf.placeholder(dtype=tf.float32, shape=[None, self.context_frames, 201], name='noisecontextph'),
                       tf.placeholder(dtype=tf.float32, shape=[None, self.context_frames, 201], name='cleancontextph'),
                       tf.placeholder(dtype=tf.int32, shape=[None], name='locationph'),
                       tf.placeholder(dtype=tf.string, shape=[None], name='cleanpathph'),
                       tf.placeholder(dtype=tf.string, shape=[None], name='noisepathph'),
                       tf.placeholder(dtype=tf.int32, shape=[None], name='snrph')]
        return self.ph



if __name__ == '__main__':
    g = tf.Graph()
    with g.as_default():
        reader = read_seeds(name='train', queuesize=200, min_after_dequeue=100, nthreads=16)
        with tf.device('/cpu:0'):
            reader.preparations()

        print('starting queue runners')
        sess = tf.Session(graph=g, config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        queue_stats = tf.stack([x.queue.size() for x in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)])
        feed_dict = {reader.mb: 64}

        with tf.device('/gpu:0'):
            clean, mixed, mixedph, noisecontext, cleancontext, location, cleanpath, noisepath, snr = reader.get_inputs()

        for i in range(1000):
            print(i)
            t1 = time.time()
            clean_, mixed_, mixedph_, noisecontext_, cleancontext_, location_, cleanpath_, noisepath_ , snr_ = sess.run(reader.batch, feed_dict=feed_dict)
            t2 = time.time()
            print(sess.run(queue_stats), t2 - t1)

