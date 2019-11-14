########################################################################################################################
#                                          N-HANS speech separator                                                     #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#   Description:      N-HANS contains two systems, speech denoiser and speech separator. This is the first part.       #
#   Authors:          Shuo Liu, Gil Keren, Bjoern Schuller                                                             #
#   Affiliation:      ZD.B Chair of Embedded Intelligence for Health Care and Wellbeing, University of Augsburg (UAU)  #
#   Version:          1.0                                                                                              #
#   Last Update:      Nov. 14, 2019                                                                                    #
#   Dependence Files: reader.py  blocks.py                                                                             #
#   Contact:          shuo.liu@informatik.uni-augburg.de                                                               #
########################################################################################################################

# Import from standard libraries
from __future__ import division, absolute_import, print_function
import tensorflow as tf, os, time, numpy as np, scipy.io.wavfile
from datetime import datetime
from os.path import join
from blocks import dense, conv2d, batch_norm, flatten
from reader import read_seeds

######################################     Global FLAGS and Training FLAGS     #########################################
# General flags
'''
During Training     : eval_before_train => False, eval_after_train => True
During Test         : eval_before_train => True, eval_after_train => False
restore_path        : the path to restore a trained model
checkpoints         : checkpoints directory
summaries           : summaries directory
dump_results        : directory of intermediate output of models
eval_every          : evaluation every "eval_every" batches
train_monitor_every : monitor training process every "train_monitor_every" batches
'''
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('eval_before_training', False, '')
tf.app.flags.DEFINE_boolean('eval_after_training', True, '')
tf.app.flags.DEFINE_integer('checkpoints_to_keep', 1000000, '')
tf.app.flags.DEFINE_string('restore_path', '', '')
tf.app.flags.DEFINE_string('model_name', 'nhans', '')
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints', '')
tf.app.flags.DEFINE_string('summaries_dir', './summaries', '')
tf.app.flags.DEFINE_string('dump_results', './dump', '')
tf.app.flags.DEFINE_integer('eval_every', 5000, '')
tf.app.flags.DEFINE_integer('train_monitor_every', 1000, '')
  
# Training and model flags
'''
batches       : maximum batches for training
alg           : optimiser
lr            : learning rate
mom           : momentum for optimiser
w_std, b_init : parameters initialisation
bn_decay      : batch normalisation decay
train_mb      : mini-batch size for training
eval_mb       : mini-batch size for valid or test
'''
tf.app.flags.DEFINE_integer('batches', 3000000, '')
tf.app.flags.DEFINE_string('alg', 'sgd', '')
tf.app.flags.DEFINE_float('lr', 0.1, '')
tf.app.flags.DEFINE_float('mom', 0.0, '')
tf.app.flags.DEFINE_float('w_std', 0.01, '')
tf.app.flags.DEFINE_float('b_init', 0.0, '')
tf.app.flags.DEFINE_float('bn_decay', 0.95, '')
tf.app.flags.DEFINE_integer('train_mb', 64, '')
tf.app.flags.DEFINE_integer('eval_mb', 100, '')

[os.system('mkdir {}'.format(dir_name)) for dir_name in ['wav_dump', 'dump', 'checkpoints', 'summaries'] if not os.path.exists(dir_name)]
########################################     Preparation of Data Reader     ############################################
global treader, tsess, tph, modelname, isess, tin

# data reader for training
def get_train_reader():
  return read_seeds('train', 40000, 6666, 16)

# data reader for validation and test
def get_eval_readers():
  return [read_seeds('valid', 5000, 0, 1)]


############################################     N-HANS Separator Model     ############################################
def model(inputs, istrain):
  clean, mixed, mixedph, noisecontext, cleancontext, location, cleanpath, noisepath, snr = inputs
  nfeat = clean.shape[2].value

  def noise_resnet_block(inputs, kernel_size, stride, n_fmaps, scope_name):
    '''
    Embedding network
    :param inputs : context input
    :return out   : embedding vector represents the context information
    '''
    # The transformation path
    path1 = conv2d(inputs, kernel_size, [1] + stride + [1], n_fmaps, FLAGS.w_std, FLAGS.b_init, False, 'SAME', scope_name + '_conv1')
    path1 = batch_norm(istrain, path1, scope_name + '_conv1')
    path1 = tf.nn.relu(path1)
    path1 = conv2d(path1, kernel_size, [1,1,1,1], n_fmaps, FLAGS.w_std, FLAGS.b_init, True, 'SAME', scope_name + '_conv2')
    
    # The identity path
    n_input_channels = inputs.shape.as_list()[3]
    if n_input_channels == n_fmaps:
      path2 = inputs
    else:
      path2 = conv2d(inputs, [1, 1], [1] + stride + [1], n_fmaps, FLAGS.w_std, FLAGS.b_init, True, 'SAME', scope_name + '_transform')
    
    # Add and return 
    assert path1.shape.as_list() == path2.shape.as_list()
    out = path1 + path2
    out = batch_norm(istrain, out, scope_name + '_addition')
    out = tf.nn.relu(out)
    return out
    
  def resnet_block(inputs, noiseemb, cleanemb, kernel_size, stride, n_fmaps, scope_name):
    '''
    Residual block to process noisy signals, with injection of embedding vectors
    :param inputs      : input feature maps
    :param noiseposemb : positive embedding vector
    :param noisenegemb : negative embedding vector
    :param n_fmaps     : number of channels
    :return out        : output feature maps
    '''
    def cont_embed(n, out_dim, scope_name):
      out = tf.constant(range(0, n), dtype=tf.float32) # [n]
      out = tf.reshape(out, [n, 1])  # [n, 1]
      out = dense(out, 50, FLAGS.w_std, 0.0, False, scope_name + '_dense1') # [n, 50]
      out = batch_norm(istrain, out, scope_name + scope_name + '_dense1')
      out = tf.nn.relu(out)
      out = dense(out, 50, FLAGS.w_std, 0.0, False, scope_name + '_dense2') # [n, 50]
      out = batch_norm(istrain, out, scope_name + scope_name + '_dense2')
      out = tf.nn.relu(out)
      out = dense(out, out_dim, 0.0, 0.0, False, scope_name + '_dense3') # [n, out_dim]
      return out
    
    def process_noise_t_f(match_to, scope_name):
      n_fmaps = match_to.shape[3].value
      # Project the noise to fit the conv
      noise_proj = dense(noiseemb, n_fmaps, 0.0, 0.0, True, scope_name + '_noise_emb') # [mb, n_fmaps]
      noise_proj = tf.expand_dims(noise_proj, 1)
      noise_proj = tf.expand_dims(noise_proj, 1) # [mb, 1, 1, n_fmaps]

      clean_proj = dense(cleanemb, n_fmaps, 0.0, 0.0, True, scope_name + '_clean_emb') # [mb, n_fmaps]
      clean_proj = tf.expand_dims(clean_proj, 1)
      clean_proj = tf.expand_dims(clean_proj, 1) # [mb, 1, 1, n_fmaps]
      
      # Get the time and frequency embedding
      ts, fs = match_to.shape[1].value, match_to.shape[2].value
      tout = cont_embed(ts, n_fmaps, scope_name + '_temb') # [ts, n_fmaps]
      tout = tf.expand_dims(tout, 1)
      tout = tf.expand_dims(tout, 0) # [1, time, 1, n_fmaps]
      fout = cont_embed(fs, n_fmaps, scope_name + '_femb') # [fs, n_fmaps]
      fout = tf.expand_dims(fout, 0)
      fout = tf.expand_dims(fout, 0) # [1, 1, freq, n_fmaps]
      
      return noise_proj, clean_proj, tout, fout
    
    # The transformation path
    path1 = conv2d(inputs, [kernel_size, kernel_size], [1, stride, stride, 1], 
                   n_fmaps, FLAGS.w_std, FLAGS.b_init, False, 
                   'SAME', scope_name + '_conv1') # [mb, time, freq, n_fmaps]
    noise_proj1, clean_proj1, tout1, fout1 = process_noise_t_f(path1, scope_name + '_conv1')
    path1 = path1 + noise_proj1 + clean_proj1 + tout1 + fout1
    path1 = batch_norm(istrain, path1, scope_name + '_conv1')
    path1 = tf.nn.relu(path1)
    path1 = conv2d(path1, [kernel_size, kernel_size], [1,1,1,1], n_fmaps, FLAGS.w_std, FLAGS.b_init, True, 'SAME', scope_name + '_conv2')
    noise_proj2, clean_proj2, tout2, fout2 = process_noise_t_f(path1, scope_name + '_conv2')
    path1 = path1 + noise_proj2 + clean_proj2 + tout2 + fout2

    # The identity path
    n_input_channels = inputs.shape.as_list()[3]
    if n_input_channels == n_fmaps:
      path2 = inputs
    else:
      path2 = conv2d(inputs, [1, 1], [1, stride, stride, 1], n_fmaps, FLAGS.w_std, FLAGS.b_init, True, 'SAME', scope_name + '_transform')
    
    # Add and return 
    assert path1.shape.as_list() == path2.shape.as_list()
    out = path1 + path2
    out = batch_norm(istrain, out, scope_name + '_addition')
    out = tf.nn.relu(out)
    return out


  # The interference speaker embedding
  with tf.variable_scope('embedding'):
    nout = None
    nout = noisecontext # [mb, noise frames, 201]
    nout = tf.expand_dims(nout, 3)
    nout = noise_resnet_block(nout, [8, 4], [3, 2], 64, 'noise_resblock1_1') # [mb, noise frames, 201, 64]
    nout = noise_resnet_block(nout, [8, 4], [3, 2], 128, 'noise_resblock2_1') # [mb, noise frames / 2, 201 / 2, 64]
    nout = noise_resnet_block(nout, [4, 4], [1, 1], 256, 'noise_resblock3_1') # [mb, noise frames / 4, 201 / 4, 64]
    nout = noise_resnet_block(nout, [4, 4], [1, 2], 512, 'noise_resblock4_1') # [mb, noise frames / 8, 201 / 8, 512]
    nout = tf.nn.avg_pool(nout, [1, nout.shape[1].value, nout.shape[2].value, 1], [1,1,1,1], 'VALID') # [mb, 1, 1, 512]
    assert nout.shape.as_list()[1:3] == [1, 1]
    noiseemb = nout[:, 0, 0, :] # [mb, 512]

  # The target speaker embedding
  with tf.variable_scope('embedding', reuse=True):
    nout = None
    nout = cleancontext # [mb, noise frames, 201]
    nout = tf.expand_dims(nout, 3)
    nout = noise_resnet_block(nout, [8, 4], [3, 2], 64, 'noise_resblock1_1') # [mb, noise frames, 201, 64]
    nout = noise_resnet_block(nout, [8, 4], [3, 2], 128, 'noise_resblock2_1') # [mb, noise frames / 2, 201 / 2, 64]
    nout = noise_resnet_block(nout, [4, 4], [1, 1], 256, 'noise_resblock3_1') # [mb, noise frames / 4, 201 / 4, 64]
    nout = noise_resnet_block(nout, [4, 4], [1, 2], 512, 'noise_resblock4_1') # [mb, noise frames / 8, 201 / 8, 512]
    nout = tf.nn.avg_pool(nout, [1, nout.shape[1].value, nout.shape[2].value, 1], [1,1,1,1], 'VALID') # [mb, 1, 1, 512]
    assert nout.shape.as_list()[1:3] == [1,1]
    cleanemb = nout[:,0,0,:] # [mb, 512]


  # Processing the mixed signal
  out = mixed # [mb, context frames, 201]
  out = tf.expand_dims(out, 3)
  out = resnet_block(out, noiseemb, cleanemb, 4, 1, 64, 'resblock1_1')
  out = resnet_block(out, noiseemb, cleanemb, 4, 1, 64, 'resblock1_2')
  out = resnet_block(out, noiseemb, cleanemb, 4, 2, 128, 'resblock2_1')
  out = resnet_block(out, noiseemb, cleanemb, 4, 1, 128, 'resblock2_2')
  out = resnet_block(out, noiseemb, cleanemb, 3, 2, 256, 'resblock3_1')
  out = resnet_block(out, noiseemb, cleanemb, 3, 1, 256, 'resblock3_2')
  out = resnet_block(out, noiseemb, cleanemb, 3, 2, 512, 'resblock4_1')
  out = resnet_block(out, noiseemb, cleanemb, 3, 1, 512, 'resblock4_2') # [mb, context frames / 8, 201 / 8, 512]

  # final layers
  out = conv2d(out, [out.shape[1].value, 1], [1, 1, 1, 1],
              512, FLAGS.w_std, FLAGS.b_init, False,
              'VALID', 'last_conv')                      # [mb, 1, 201 / 8, 512]
  out = batch_norm(istrain, out, 'last_conv')
  out = tf.nn.relu(out)
  out = flatten(out)                                     # [mb,  (201 / 8) * 512]
  out = dense(out, nfeat, 0.0, 0.0, True, 'last_dense')  # [mb, 201]
  mixed_central = mixed[:, FLAGS.window_frames // 2, :]  # [mb, 201]
  denoised = mixed_central + out                         # [mb, 201]
  
  # Loss
  se = tf.square(denoised - clean[:,0,:])                # [mb, 201]
  imp_factor = np.linspace(2, 1, nfeat, dtype=np.float32).reshape((1, nfeat))
  example_loss = tf.reduce_mean(se * tf.constant(imp_factor), axis=1)
  loss = tf.reduce_mean(example_loss)
  
  monitors = {'loss': loss}
  outputs = {'loss': example_loss, 'mixed': mixed_central, 'denoised': denoised, 'clean': clean[:,0,:],
             'mixedph': mixedph[:,0,:], 'location': location, 'cleanpath': cleanpath, 
             'noisepath': noisepath, 'snr': snr}
  return loss, monitors, outputs


############################################     Restoration from Estimated Spectrum     ###############################
def evaluate(outputs, ereader, step):
  '''
  Restoration from Estimated Spectrum
  :param outputs : estimated spectrograms
  :param ereader : evaluation data reader
  :param step    : global step
  '''
  print(ereader.name)
  
  # Print the loss
  loss = outputs['loss'].mean()
  print('loss: {}'.format(loss))
  
  # Create a summary
  summary = tf.Summary()
  summary.value.add(tag='{}_{}'.format(ereader.name, 'loss'), simple_value=loss)
  
  # Construct the tf graph for reconstruction
  g = tf.Graph()
  with g.as_default():
    stft = tf.placeholder(dtype=tf.complex64, shape=[None, 201])
    samplestf = tf.contrib.signal.inverse_stft(tf.expand_dims(stft, 0), ereader.frame_length, ereader.frame_step, ereader.frame_length)[0,:]
    sess = tf.Session()
  
  # Reconstruct some audio
  startlocations = np.where(outputs['location'] == 0)[0]
  for i in range(0, len(startlocations)):
    # Start and end location
    s = startlocations[i]
    if i == len(startlocations) - 1:
      e = len(outputs['mixed'])
    else:
      e = startlocations[i+1]
    
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
    
    # Reconstruct waveform
    mixedsamples = sess.run(samplestf, feed_dict={stft: mixedspect})
    denoisedsamples = sess.run(samplestf, feed_dict={stft: denoisedspect})
    
    # Save the waveforms
    cleanpath, noisepath, snr = outputs['cleanpath'][s], outputs['noisepath'][s], outputs['snr'][s]
    cleanpath = cleanpath.split('/')[-1][:-4]
    noisepath = noisepath.split('/')[-1][:-4]
    mixed_filename = '{}_{}_{}_{}_{}_{}.wav'.format(modelname, step, cleanpath, noisepath, snr, 'mixed')
    denoised_filename = '{}_{}_{}_{}_{}_{}.wav'.format(modelname, step, cleanpath, noisepath, snr, 'denoised')
    scipy.io.wavfile.write(os.path.join(FLAGS.wav_dump_folder, mixed_filename), 16000, mixedsamples)
    scipy.io.wavfile.write(os.path.join(FLAGS.wav_dump_folder, denoised_filename), 16000, denoisedsamples)


#####################################     Main Loop for Training and Evaluation with Monitors    #######################
def main_loop():
  global treader, tsess, tph, modelname, isess, tin
  
  # Print FLAGS
  _ = FLAGS.lr
  print('----------------------------- FLAGS VALUES --------------------------------')
  for k in sorted(FLAGS.__flags.keys()):
    print('{}: {}'.format(k, getattr(FLAGS, k)))
  print('')
  
  
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

    isess = tf.Session(graph=ig, config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))


  # Training model settings
  tg = tf.Graph()
  with tg.as_default():
    # Session
    tsess = tf.Session(graph=tg, config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))

    # The model
    with tf.device('/gpu:0'):
      tin = treader.get_inputs()
      loss, monitors, _ = model(tin, True)
      # The global step
      global_step = tf.Variable(0, trainable=False)
      # Optimization
      if FLAGS.alg == 'sgd':
        opt = tf.train.GradientDescentOptimizer(FLAGS.lr)
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
      grads_and_vars = opt.compute_gradients(loss, tf.trainable_variables())
      train_op = opt.apply_gradients(grads_and_vars)
      print('#trainable variables: {}'.format(sum([x.get_shape().num_elements() for x in tf.trainable_variables()])))
      print('#non-trainable variables: {}'.format(sum([x.get_shape().num_elements() for x in set(tf.global_variables()) - set(tf.trainable_variables())])))
      assert not tf.local_variables() # Should be empty always, otherwise change the lines above
    
    # Save, restore and init
    tsaver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.checkpoints_to_keep)
    if FLAGS.restore_path:
      # Restore the variables values
      print('Restoring model from {}'.format(FLAGS.restore_path))
      restore_saver = tf.train.Saver(tf.global_variables())
      restore_saver.restore(tsess, FLAGS.restore_path)
    else:
      tsess.run(tf.global_variables_initializer())


  # Evaluation model settings
  eg = tf.Graph()
  with eg.as_default():
    esess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    with tf.device('/gpu:0'):
      ein = ereaders[0].get_inputs()
      _, _, outputs = model(ein, False)
    esaver = tf.train.Saver(tf.global_variables())

  
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
        isess.run(tf.local_variables_initializer())
        icoord = tf.train.Coordinator()
        ithreads = tf.train.start_queue_runners(sess=isess, coord=icoord, start=False)
        queue_stats = tf.stack([x.queue.size() for x in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)])
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
                isess.run(tf.local_variables_initializer())
                nowcoord = tf.train.Coordinator()
                nowthreads = []
                qr1 = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)[-3]
                qr2 = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)[-2]
                qr3 = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)[-1]
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
              np.save(join(FLAGS.dump_results, '{}_{}_{}_{}'.format(modelname, ereader.name, tstep, k)), aggregators[k])

          # Evaluate
          evaluate(aggregators, ereader, global_step)

        # Print the bottom part
        print('-----------------------------------------------------')
        print('')

    # Eval before training
    if FLAGS.eval_before_training:
      save_and_eval()
    
    # Training loop
    while tstep < FLAGS.batches and not icoord.should_stop():
      # Running the step
      input_feed_dict = {treader.mb: FLAGS.train_mb} 
      #t1 = time.time()
      batch = isess.run(treader.batch, feed_dict=input_feed_dict)
      #t2 = time.time()
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
          print('{}: {}'.format(k, aggregators[k]))                            # loss
        end = time.time() 
        print('queues stats: {}'.format(isess.run(queue_stats)))
        print('seconds elapsed: {}'.format(end - start)) 
        start = end
        # Add summaries and reset aggregators
        summary = tf.Summary()
        summary.value.add(tag='{}_{}'.format(treader.name, 'loss'), simple_value=aggregators['loss'])
        #treader.swriter.add_summary(summary, tstep)
        aggregators = {k: 0.0 for k in monitors.keys()}
        print('---------------------------------------------------------')
        print('')

      # Save and eval
      if tstep % FLAGS.eval_every == 0:
        save_and_eval()
        
    # Eval after training
    if FLAGS.eval_after_training:
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
