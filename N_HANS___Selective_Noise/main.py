########################################################################################################################
#                                          N-HANS speech denoiser                                                      #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#   Discription:      N-HANS contains two parts, speech denoiser and speech separator.
#   Authors:          Shuo Liu, Gil Keren, Bjoern Schuller                                                             #
#   Afficiation:      ZD.B Chair of Embedded Intelligence for Health Care and Wellbeing, University of Augsburg (UAU)  #
#   Date and Time:    May. 04, 2020                                                                                    #
#   Modified:         xxx                                                                                              #
#   Version:          1.5                                                                                              #
#   Dependence Files: reader.py  blocks.py                                                                             #
#   Contact:          shuo.liu@informatik.uni-augburg.de                                                               #
########################################################################################################################

# Import from standard libraries
from __future__ import division, absolute_import, print_function
import os, time, numpy as np, scipy.io.wavfile, functools
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
if int(tf.__version__.split('.')[0]) == 2:
    import tensorflow.compat.v1 as tf
    tf.compat.v1.disable_v2_behavior()
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from blocks import dense, conv2d, batch_norm, flatten
from reader import read_seeds
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# General flags
"""
During Training     : eval_before_train => False, eval_after_train => True
During Test         : eval_before_train => True, eval_after_train => False
restore_path        : the path to restore a trained model
checkpoints         : checkpoints directory
summaries           : summaries directory
dump_results        : directory of intermediate output of models
eval_every          : evaluation every "eval_every" batches
train_monitor_every : monitor training process every "train_monitor_every" batches
"""

FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_boolean('eval_before_training', False, '')
tf.compat.v1.flags.DEFINE_boolean('eval_after_training', True, '')
tf.compat.v1.flags.DEFINE_integer('checkpoints_to_keep', 1000000, '')
tf.compat.v1.flags.DEFINE_string('restore_path', '', '')
tf.compat.v1.flags.DEFINE_string('model_name', 'nhans', '')
tf.compat.v1.flags.DEFINE_string('checkpoint_dir', './checkpoints', '')
tf.compat.v1.flags.DEFINE_string('summaries_dir', './summaries', '')
tf.compat.v1.flags.DEFINE_string('dump_results', './dump', '')
tf.compat.v1.flags.DEFINE_integer('eval_every', 5000, '')                                 # 3
tf.compat.v1.flags.DEFINE_integer('train_monitor_every', 1000, '')                        # 3

# Training flags
"""
batches       : maximum batches for training
alg           : optimiser
lr            : learning rate
mom           : momentum for optimiser
w_std, b_init : parameters initialisation
bn_decay      : batch normalisation decay
train_mb      : mini-batch size for training
eval_mb       : mini-batch size for valid or test
"""
tf.compat.v1.flags.DEFINE_integer('batches', 3000000, '')
tf.compat.v1.flags.DEFINE_string('alg', 'sgd', '')
tf.compat.v1.flags.DEFINE_float('lr', 0.001, '')
tf.compat.v1.flags.DEFINE_float('mom', 0.0, '')
tf.compat.v1.flags.DEFINE_float('w_std', 0.01, '')
tf.compat.v1.flags.DEFINE_float('b_init', 0.0, '')
tf.compat.v1.flags.DEFINE_float('bn_decay', 0.95, '')
tf.compat.v1.flags.DEFINE_integer('train_mb', 64, '')                                   # 64
tf.compat.v1.flags.DEFINE_integer('eval_mb', 100, '')                                   # 100

[os.system('mkdir {}'.format(dir_name))
 for dir_name in ['wav_dump', 'dump', 'checkpoints', 'summaries'] if not os.path.exists(dir_name)]

"""
Preparation of Data Reader
"""


global treader, tsess, tph, modelname, isess, tin


def get_train_reader():
    return read_seeds('train', 40000, 6666, 16)                                          # 10, 1, 2


def get_eval_readers():
    return [read_seeds('valid', 5000, 0, 1)]                                        # 10, 0, 1


""" 
N-HANS Denoiser Model
"""


def model(inputs, istrain):
    target, mixed, mixedph, targetph, pos, posph, neg, negph, noiseposcontext, noisenegcontext, location, cleanpath, noisepospath, noisenegpath, snr_pos, snr_neg = inputs
    nfeat = target.shape[2].value

    def noise_resnet_block(inputs, kernel_size, stride, n_fmaps, scope_name):
        # The transformation path
        path1 = conv2d(inputs, kernel_size, [1] + stride + [1], n_fmaps, FLAGS.w_std, FLAGS.b_init, False, 'SAME',
                       scope_name + '_conv1')
        path1 = batch_norm(istrain, path1, scope_name + '_conv1')
        path1 = tf.nn.relu(path1)
        path1 = conv2d(path1, kernel_size, [1, 1, 1, 1], n_fmaps, FLAGS.w_std, FLAGS.b_init, True, 'SAME',
                       scope_name + '_conv2')

        # The identity path
        n_input_channels = inputs.shape.as_list()[3]
        if n_input_channels == n_fmaps:
            path2 = inputs
        else:
            path2 = conv2d(inputs, [1, 1], [1] + stride + [1], n_fmaps, FLAGS.w_std, FLAGS.b_init, True, 'SAME',
                           scope_name + '_transform')

        # Add and return
        assert path1.shape.as_list() == path2.shape.as_list()
        out = path1 + path2
        out = batch_norm(istrain, out, scope_name + '_addition')
        out = tf.nn.relu(out)
        return out

    def resnet_block(inputs, noiseposemb, noisenegemb, kernel_size, stride, n_fmaps, scope_name):
        def cont_embed(n, out_dim, scope_name):
            out = tf.constant(list(range(0, n)), dtype=tf.float32)  # [n]
            out = tf.reshape(out, [n, 1])  # [n, 1]
            out = dense(out, 50, FLAGS.w_std, 0.0, False, scope_name + '_dense1')  # [n, 50]
            out = batch_norm(istrain, out, scope_name + scope_name + '_dense1')
            out = tf.nn.relu(out)
            out = dense(out, 50, FLAGS.w_std, 0.0, False, scope_name + '_dense2')  # [n, 50]
            out = batch_norm(istrain, out, scope_name + scope_name + '_dense2')
            out = tf.nn.relu(out)
            out = dense(out, out_dim, 0.0, 0.0, False, scope_name + '_dense3')  # [n, out_dim]
            return out

        def process_noise_t_f(match_to, scope_name):
            n_fmaps = match_to.shape[3].value
            # Project the noise to fit the conv
            noisepos_proj = dense(noiseposemb, n_fmaps, 0.0, 0.0, True, scope_name + '_noise_pos_emb')  # [mb, n_fmaps]
            noisepos_proj = tf.expand_dims(noisepos_proj, 1)
            noisepos_proj = tf.expand_dims(noisepos_proj, 1)  # [mb, 1, 1, n_fmaps]

            noiseneg_proj = dense(noisenegemb, n_fmaps, 0.0, 0.0, True, scope_name + '_noise_neg_emb')  # [mb, n_fmaps]
            noiseneg_proj = tf.expand_dims(noiseneg_proj, 1)
            noiseneg_proj = tf.expand_dims(noiseneg_proj, 1)  # [mb, 1, 1, n_fmaps]

            # Get the time and frequency embedding
            ts, fs = match_to.shape[1].value, match_to.shape[2].value
            tout = cont_embed(ts, n_fmaps, scope_name + '_temb')  # [ts, n_fmaps]
            tout = tf.expand_dims(tout, 1)
            tout = tf.expand_dims(tout, 0)  # [1, time, 1, n_fmaps]
            fout = cont_embed(fs, n_fmaps, scope_name + '_femb')  # [fs, n_fmaps]
            fout = tf.expand_dims(fout, 0)
            fout = tf.expand_dims(fout, 0)  # [1, 1, freq, n_fmaps]

            return noisepos_proj, noiseneg_proj, tout, fout

        # The transformation path
        path1 = conv2d(inputs, [kernel_size, kernel_size], [1, stride, stride, 1],
                       n_fmaps, FLAGS.w_std, FLAGS.b_init, False,
                       'SAME', scope_name + '_conv1')  # [mb, time, freq, n_fmaps]
        noisepos_proj1, noiseneg_proj1, tout1, fout1 = process_noise_t_f(path1, scope_name + '_conv1')
        path1 = path1 + noisepos_proj1 + noiseneg_proj1 + tout1 + fout1
        path1 = batch_norm(istrain, path1, scope_name + '_conv1')
        path1 = tf.nn.relu(path1)
        path1 = conv2d(path1, [kernel_size, kernel_size], [1, 1, 1, 1], n_fmaps, FLAGS.w_std, FLAGS.b_init, True,
                       'SAME', scope_name + '_conv2')
        noisepos_proj2, noiseneg_proj2, tout2, fout2 = process_noise_t_f(path1, scope_name + '_conv2')
        path1 = path1 + noisepos_proj2 + noiseneg_proj2 + tout2 + fout2

        # The identity path
        n_input_channels = inputs.shape.as_list()[3]
        if n_input_channels == n_fmaps:
            path2 = inputs
        else:
            path2 = conv2d(inputs, [1, 1], [1, stride, stride, 1], n_fmaps, FLAGS.w_std, FLAGS.b_init, True, 'SAME',
                           scope_name + '_transform')

        # Add and return
        assert path1.shape.as_list() == path2.shape.as_list()
        out = path1 + path2
        out = batch_norm(istrain, out, scope_name + '_addition')
        out = tf.nn.relu(out)
        return out

    # The noise embedding
    with tf.compat.v1.variable_scope('embedding'):
        nout = None
        nout = noiseposcontext  # [mb, noise frames, 201]
        nout = tf.expand_dims(nout, 3)
        nout = noise_resnet_block(nout, [8, 4], [3, 2], 64, 'noise_resblock1_1')  # [mb, noise frames, 201, 64]
        nout = noise_resnet_block(nout, [8, 4], [3, 2], 128, 'noise_resblock2_1')  # [mb, noise frames / 2, 201 / 2, 64]
        nout = noise_resnet_block(nout, [4, 4], [1, 1], 256, 'noise_resblock3_1')  # [mb, noise frames / 4, 201 / 4, 64]
        nout = noise_resnet_block(nout, [4, 4], [1, 2], 512,
                                  'noise_resblock4_1')  # [mb, noise frames / 8, 201 / 8, 512]
        nout = tf.nn.avg_pool2d(nout, [1, nout.shape[1].value, nout.shape[2].value, 1], [1, 1, 1, 1],
                                'VALID')  # [mb, 1, 1, 512]
        assert nout.shape.as_list()[1:3] == [1, 1]
        noiseposemb = nout[:, 0, 0, :]  # [mb, 512]

    with tf.compat.v1.variable_scope('embedding', reuse=True):
        nout = None
        nout = noisenegcontext  # [mb, noise frames, 201]
        nout = tf.expand_dims(nout, 3)
        nout = noise_resnet_block(nout, [8, 4], [3, 2], 64, 'noise_resblock1_1')  # [mb, noise frames, 201, 64]
        nout = noise_resnet_block(nout, [8, 4], [3, 2], 128, 'noise_resblock2_1')  # [mb, noise frames / 2, 201 / 2, 64]
        nout = noise_resnet_block(nout, [4, 4], [1, 1], 256, 'noise_resblock3_1')  # [mb, noise frames / 4, 201 / 4, 64]
        nout = noise_resnet_block(nout, [4, 4], [1, 2], 512,
                                  'noise_resblock4_1')  # [mb, noise frames / 8, 201 / 8, 512]
        nout = tf.nn.avg_pool2d(nout, [1, nout.shape[1].value, nout.shape[2].value, 1], [1, 1, 1, 1],
                                'VALID')  # [mb, 1, 1, 512]
        assert nout.shape.as_list()[1:3] == [1, 1]
        noisenegemb = nout[:, 0, 0, :]  # [mb, 512]

    # Processing the mixed signal
    out = mixed  # [mb, context frames, 201]
    out = tf.expand_dims(out, 3)
    out = resnet_block(out, noiseposemb, noisenegemb, 4, 1, 64, 'resblock1_1')
    out = resnet_block(out, noiseposemb, noisenegemb, 4, 1, 64, 'resblock1_2')
    out = resnet_block(out, noiseposemb, noisenegemb, 4, 2, 128, 'resblock2_1')
    out = resnet_block(out, noiseposemb, noisenegemb, 4, 1, 128, 'resblock2_2')
    out = resnet_block(out, noiseposemb, noisenegemb, 3, 2, 256, 'resblock3_1')
    out = resnet_block(out, noiseposemb, noisenegemb, 3, 1, 256, 'resblock3_2')
    out = resnet_block(out, noiseposemb, noisenegemb, 3, 2, 512, 'resblock4_1')
    out = resnet_block(out, noiseposemb, noisenegemb, 3, 1, 512,
                       'resblock4_2')  # [mb, context frames / 8, 201 / 8, 512]

    # final layers
    out = conv2d(out, [out.shape[1].value, 1], [1, 1, 1, 1],
                 512, FLAGS.w_std, FLAGS.b_init, False,
                 'VALID', 'last_conv')  # [mb, 1, 201 / 8, 512]
    out = batch_norm(istrain, out, 'last_conv')
    out = tf.nn.relu(out)
    out = flatten(out)  # [mb,  (201 / 8) * 512]
    out = dense(out, nfeat, 0.0, 0.0, True, 'last_dense')  # [mb, 201]
    mixed_central = mixed[:, FLAGS.window_frames // 2, :]  # [mb, 201]
    pos_central = pos[:, FLAGS.window_frames // 2, :]  # [mb, 201]
    neg_central = neg[:, FLAGS.window_frames // 2, :]  # [mb, 201]
    denoised = mixed_central + out  # [mb, 201]

    # Loss
    se = tf.square(denoised - target[:, 0, :])  # [mb, 201]
    imp_factor = np.linspace(2, 1, nfeat, dtype=np.float32).reshape((1, nfeat))
    example_loss = tf.reduce_mean(se * tf.constant(imp_factor), axis=1)
    loss = tf.reduce_mean(example_loss)

    monitors = {'loss': loss}
    outputs = {'loss': example_loss, 'mixed': mixed_central, 'denoised': denoised, 'target': target[:, 0, :],
               'mixedph': mixedph[:, 0, :], 'targetph': targetph[:, 0, :], 'pos': pos_central, 'neg': neg_central,
               'posph': posph[:, 0, :], 'negph': negph[:, 0, :], 'location': location, 'cleanpath': cleanpath,
               'noisepospath': noisepospath, 'noisenegpath': noisenegpath, 'snr_pos': snr_pos,
               'snr_neg': snr_neg}  # , 'temp_wav': temp_wav}
    return loss, monitors, outputs


"""
Restoration from Estimated Spectrum
"""


def evaluate(outputs, ereader, step):
    print(ereader.name)

    # Print the loss
    loss = outputs['loss'].mean()
    print('loss: {}'.format(loss))

    # Create a summary
    summary = tf.compat.v1.Summary()
    summary.value.add(tag='{}_{}'.format(ereader.name, 'loss'), simple_value=loss)
    # ereader.writer.add_summary(summary, step)

    # Construct the tf graph for reconstruction
    g = tf.Graph()
    with g.as_default():
        num_fea = int(FLAGS.Fs * 0.025) / 2 + 1
        stft = tf.compat.v1.placeholder(dtype=tf.complex64, shape=[None, num_fea])
        samplestf = tf.signal.inverse_stft(tf.expand_dims(stft, 0), ereader.frame_length, ereader.frame_step,
                                           ereader.frame_length, window_fn=tf.signal.inverse_stft_window_fn(ereader.frame_step, forward_window_fn=functools.partial(tf.signal.hann_window, periodic=True)))[0, :]
        sess = tf.compat.v1.Session()

    # Reconstruct some audio
    startlocations = np.where(outputs['location'] == 0)[0]
    for i in range(0, len(startlocations)):
        # Start and end location
        s = startlocations[i]
        if i == len(startlocations) - 1:
            e = len(outputs['mixed'])
        else:
            e = startlocations[i + 1]

        # Collect frames
        mixedmagframes = outputs['mixed'][s:e]
        mixedphframes = outputs['mixedph'][s:e]
        denoisedmagframes = outputs['denoised'][s:e]

        # Undo the log operation
        mixedmagframes = np.exp(mixedmagframes)
        denoisedmagframes = np.exp(denoisedmagframes)

        # Reconstruct complex spectrogram
        mixedspect = mixedmagframes * (np.exp(1j * mixedphframes))
        denoisedspect = denoisedmagframes * (np.exp(1j * mixedphframes))

        # collect other references
        targetmagframes = outputs['target'][s:e]
        targetphframes = outputs['targetph'][s:e]
        posmagframes = outputs['pos'][s:e]
        posphframes = outputs['posph'][s:e]
        negmagframes = outputs['neg'][s:e]
        negphframes = outputs['negph'][s:e]

        targetmagframes = np.exp(targetmagframes)
        targetspect = targetmagframes * (np.exp(1j * targetphframes))
        posmagframes = np.exp(posmagframes)
        posspect = posmagframes * (np.exp(1j * posphframes))
        negmagframes = np.exp(negmagframes)
        negspect = negmagframes * (np.exp(1j * negphframes))

        # Reconstruct waveform
        mixedsamples = sess.run(samplestf, feed_dict={stft: mixedspect})
        denoisedsamples = sess.run(samplestf, feed_dict={stft: denoisedspect})
        targetsamples = sess.run(samplestf, feed_dict={stft: targetspect})
        possamples = sess.run(samplestf, feed_dict={stft: posspect})
        negsamples = sess.run(samplestf, feed_dict={stft: negspect})

        # Save the waveforms
        cleanpath, noisepospath, noisenegpath, snr_pos, snr_neg = outputs['cleanpath'][s], outputs['noisepospath'][s], \
                                                                  outputs['noisenegpath'][s], outputs['snr_pos'][s], \
                                                                  outputs['snr_neg'][s]
        cleanpath, noisepospath, noisenegpath = cleanpath.decode('utf-8'), noisepospath.decode(
            'utf-8'), noisenegpath.decode('utf-8')
        cleanpath = cleanpath.split('/')[-1][:-4]
        noisepospath = noisepospath.split('/')[-1][:-4]
        noisenegpath = noisenegpath.split('/')[-1][:-4]
        mixed_filename = '{}_{}_{}_{}_{}_{}_{}_{}.wav'.format(modelname, step, cleanpath, noisepospath, noisenegpath,
                                                              snr_pos, snr_neg, 'mixed')
        denoised_filename = '{}_{}_{}_{}_{}_{}_{}_{}.wav'.format(modelname, step, cleanpath, noisepospath, noisenegpath,
                                                                 snr_pos, snr_neg, 'denoised')
        pos_filename = '{}_{}_{}_{}_{}_{}_{}_{}.wav'.format(modelname, step, cleanpath, noisepospath, noisenegpath,
                                                            snr_pos, snr_neg, 'posNoise')
        neg_filename = '{}_{}_{}_{}_{}_{}_{}_{}.wav'.format(modelname, step, cleanpath, noisepospath, noisenegpath,
                                                            snr_pos, snr_neg, 'negNoise')
        target_filename = '{}_{}_{}_{}_{}_{}_{}_{}.wav'.format(modelname, step, cleanpath, noisepospath, noisenegpath,
                                                               snr_pos, snr_neg, 'target')
        scipy.io.wavfile.write(os.path.join(FLAGS.wav_dump_folder, mixed_filename), FLAGS.Fs, mixedsamples)
        scipy.io.wavfile.write(os.path.join(FLAGS.wav_dump_folder, denoised_filename), FLAGS.Fs, denoisedsamples)
        scipy.io.wavfile.write(os.path.join(FLAGS.wav_dump_folder, target_filename), FLAGS.Fs, targetsamples)
        scipy.io.wavfile.write(os.path.join(FLAGS.wav_dump_folder, pos_filename), FLAGS.Fs, possamples)
        scipy.io.wavfile.write(os.path.join(FLAGS.wav_dump_folder, neg_filename), FLAGS.Fs, negsamples)


"""
Main Loop for Training and Evaluation with Monitors
"""


def main_loop():
    global treader, tsess, tph, modelname, isess, tin

    # Print FLAGS
    _ = FLAGS.lr
    print('----------------------------- FLAGS VALUES --------------------------------')
    for k in sorted(FLAGS.__flags.keys()):
        print('{}: {}'.format(k, getattr(FLAGS, k)))

    # Print General model information
    print('----------------------- DATA LOADING, MODEL PREPARING -------------------------')
    time_stamp = str(datetime.now()).replace(':', '-').replace(' ', '_')
    modelname = FLAGS.model_name if FLAGS.model_name else str(time_stamp)
    print('model_name: {}'.format(modelname))

    ig = tf.Graph()
    with ig.as_default():
        with tf.device('/cpu:0'):
            # Preparing the reader
            treader = get_train_reader()
            treader.preparations()

            # Readers
            ereaders = get_eval_readers()
            # for ereader in ereaders:
            #   ereader.preparations()

        isess = tf.compat.v1.Session(graph=ig, config=tf.compat.v1.ConfigProto(log_device_placement=False,
                                                                               allow_soft_placement=True))

    # Training model settings
    tg = tf.Graph()
    with tg.as_default():
        # Session
        tsess = tf.compat.v1.Session(graph=tg, config=tf.compat.v1.ConfigProto(log_device_placement=False,
                                                                               allow_soft_placement=True))

        # The model
        with tf.device('/gpu:0'):
            tin = treader.get_inputs()
            loss, monitors, _ = model(tin, True)
            # The global step
            # global_step = tf.Variable(0, trainable=False)
            # Optimization
            if FLAGS.alg == 'sgd':
                opt = tf.compat.v1.train.GradientDescentOptimizer(FLAGS.lr)
            elif FLAGS.alg == 'momentum':
                opt = tf.train.MomentumOptimizer(FLAGS.lr, FLAGS.mom)
            elif FLAGS.alg == 'rmsprop':
                opt = tf.train.RMSPropOptimizer(FLAGS.lr, momentum=FLAGS.mom)
            elif FLAGS.alg == 'adadelta':
                opt = tf.train.AdadeltaOptimizer(FLAGS.lr)
            elif FLAGS.alg == 'adagrad':
                opt = tf.train.AdagradOptimizer(FLAGS.lr)
            elif FLAGS.alg == 'adam':
                opt = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
            grads_and_vars = opt.compute_gradients(loss, tf.compat.v1.trainable_variables())
            train_op = opt.apply_gradients(grads_and_vars)
            print('#trainable variables: {}'.format(
                sum([x.get_shape().num_elements() for x in tf.compat.v1.trainable_variables()])))
            print('#non-trainable variables: {}'.format(sum([x.get_shape().num_elements() for x in
                                                             set(tf.compat.v1.global_variables()) - set(
                                                                 tf.compat.v1.trainable_variables())])))
            assert not tf.compat.v1.local_variables()  # Should be empty always, otherwise change the lines above

        # Save, restore and init
        tsaver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=FLAGS.checkpoints_to_keep)
        if FLAGS.restore_path:
            # Restore the variables values
            print('Restoring model from {}'.format(FLAGS.restore_path))
            restore_saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
            restore_saver.restore(tsess, FLAGS.restore_path)
        else:
            tsess.run(tf.compat.v1.global_variables_initializer())

    # Evaluation model settings
    eg = tf.Graph()
    with eg.as_default():
        esess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
        with tf.device('/gpu:0'):
            ein = ereaders[0].get_inputs()
            _, _, outputs = model(ein, False)
        esaver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

    # Training loop
    print('--------------------------------- TRAINING! ------------------------------------')

    # Init aggregators and names
    aggregators = {k: np.zeros(shape=monitors[k].shape.as_list()) for k in monitors.keys()}

    # Operations that need to be executed in every training step
    step_ops = {'train_op': train_op}
    step_ops.update(monitors)

    # Measure time
    start = time.time()

    # The step
    tstep = 0

    try:
        # Start queue runners
        with ig.as_default():
            with tf.device('/cpu:0'):
                isess.run(tf.compat.v1.local_variables_initializer())
                icoord = tf.train.Coordinator()
                ithreads = tf.train.start_queue_runners(sess=isess, coord=icoord, start=False)
                queue_stats = tf.stack(
                    [x.queue.size() for x in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.QUEUE_RUNNERS)])
                for t in ithreads:
                    t.start()

        def save_and_eval():
            # Saving and restoring
            print('Saving and restoring the model')
            tsaver.save(tsess,
                        os.path.join(FLAGS.checkpoint_dir, modelname),
                        global_step=tstep,
                        latest_filename=modelname)
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir, latest_filename=modelname)
            if ckpt and ckpt.model_checkpoint_path:
                print('Restoring from file: {}'.format(ckpt.model_checkpoint_path))
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return
            esaver.restore(esess, ckpt.model_checkpoint_path)

            # Print the header
            print('----------------- TEST MONITOR ----------------------')

            for ereader in ereaders:
                # Initialize aggregators
                aggregators = {k: [] for k in outputs.keys()}

                # Build the graph again
                try:
                    with ig.as_default():
                        with tf.device('/cpu:0'):
                            ereader.preparations()
                            isess.run(tf.compat.v1.local_variables_initializer())
                            nowcoord = tf.train.Coordinator()
                            nowthreads = []
                            qr1 = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.QUEUE_RUNNERS)[-3]
                            qr2 = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.QUEUE_RUNNERS)[-2]
                            qr3 = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.QUEUE_RUNNERS)[-1]
                            nowthreads.extend(qr1.create_threads(isess, coord=nowcoord, daemon=True, start=True))
                            nowthreads.extend(qr2.create_threads(isess, coord=nowcoord, daemon=True, start=True))
                            nowthreads.extend(qr3.create_threads(isess, coord=nowcoord, daemon=True, start=True))

                    i = 0
                    while True:
                        i += 1

                        input_feed_dict = {ereader.mb: FLAGS.eval_mb}
                        try:
                            batch = isess.run(ereader.batch, feed_dict=input_feed_dict)

                        except tf.errors.OutOfRangeError:
                            break
                        # QUICK_MODE = True
                        # if QUICK_MODE and i == 2:
                        #     break
                        model_feed_dict = {a: b for (a, b) in zip(ein, batch)}
                        results = esess.run(outputs, model_feed_dict)

                        # concatenation aggregation of certain channels
                        for k in results.keys():
                            aggregators[k].append(results[k])
                finally:
                    nowcoord.request_stop()
                    nowcoord.join(nowthreads)

                # Concatenate results over different batches
                for k in results.keys():
                    aggregators[k] = np.concatenate(aggregators[k])

                # Dump activations
                if FLAGS.dump_results:
                    for k in results.keys():
                        np.save(
                            os.path.join(FLAGS.dump_results, '{}_{}_{}_{}'.format(modelname, ereader.name, tstep, k)),
                            aggregators[k])

                # Evaluate ##### delete global_step
                global_step = 0
                evaluate(aggregators, ereader, global_step)

            # Print the bottom part
            print('-----------------------------------------------------')

        # Eval before training
        if FLAGS.eval_before_training:
            print('processing eval before training')
            save_and_eval()

        # Training loop
        while tstep < FLAGS.batches and not icoord.should_stop():
            # Running the step
            input_feed_dict = {treader.mb: FLAGS.train_mb}
            # t1 = time.time()
            batch = isess.run(treader.batch, feed_dict=input_feed_dict)
            # t2 = time.time()
            model_feed_dict = {a: b for (a, b) in zip(tin, batch)}
            results = tsess.run(step_ops, model_feed_dict)

            # Aggregate the train monitoring channels
            for key in monitors.keys():
                aggregators[key] += results[key]

            # Increase the step. 'step' always contains the number of gradient updates that were performed.
            tstep += 1

            # Train monitoring
            if tstep % FLAGS.train_monitor_every == 0:
                # Do some printing
                print('----- TRAIN MONITOR AFTER ANOTHER {} BATCHES ------------'.format(FLAGS.train_monitor_every))
                print('step number: {}'.format(tstep))
                for k in sorted(aggregators.keys()):
                    aggregators[k] = aggregators[k] / float(FLAGS.train_monitor_every)
                    print('{}: {}'.format(k, aggregators[k]))  # loss
                end = time.time()
                print('queues stats: {}'.format(isess.run(queue_stats)))
                print('seconds elapsed: {}'.format(end - start))
                start = end
                # Add summaries and reset aggregators
                summary = tf.compat.v1.Summary()
                summary.value.add(tag='{}_{}'.format(treader.name, 'loss'), simple_value=aggregators['loss'])
                # treader.swriter.add_summary(summary, tstep)
                aggregators = {k: 0.0 for k in monitors.keys()}
                print('---------------------------------------------------------')

            # Save and eval
            if tstep % FLAGS.eval_every == 0:
                save_and_eval()

        # Eval after training
        if FLAGS.eval_after_training:
            print('processing eval after training')
            save_and_eval()

    finally:
        icoord.request_stop()
        time.sleep(1)
        isess.close()
        tsess.close()
        esess.close()
        icoord.join(ithreads)


if __name__ == '__main__':
    main_loop()
