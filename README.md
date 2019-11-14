__N-HANS__ is a Python toolkit for in-the-wild speech enhancement, including speech, music, and general audio denoising, separation, and selective noise or source suppression. The functionalities are realised based on two neural network models sharing the same architecture, but trained separately. The models are comprised of stacks of residual blocks, each conditioned on additional speech or environmental noise recordings for adapting to different unseen speakers or environments in real life. 

In addition to a Python API, a command line interface is provided to researchers and developers:
                                    __pip install N-HANS__.

## Citation
If you use N-HANS or any code from N-HANS in your research work, you are kindly asked to acknowledge the use of N-HANS in your publications.

____________ publication to be added (archive or JMLR) ________________


## Prerequisites
* Python2.7
#### Python Dependencies
* numpy
* scipy
* six
* tensorflow or tensforflow-gpu 1.14.0

## Usage
### Loading Models
After __pip install N-HANS__, users are expexted to create a workspace for speech denoising or separation task, and then use __load_denoiser__ or __load_separator__ to download trained models and audio examples into the workspace.

### Applying N-HANS
#### Commands
| Task | Command | Discription |
|---|---|---|
|__speech denoising__| __nhans_denoiser__ __--input__ noisy.wav &nbsp;&nbsp;&nbsp;&nbsp; __--neg__=noise.wav | __--neg__ implicates the environmental noise |  
|__selective noise suppresion__| __nhans_denoiser__ __--input__ noisy.wav __--pos__=preserve.wav __--neg__=suppress.wav | __--pos__ implicates the noise to keep &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; __--neg__ hints the noise to suppress |
|__speech separation__| __nhans_separator__ __--mixed.wav__ __--pos__=target.wav  __--neg__=interference.wav | __--pos__ implicates the target speaker &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  __--neg__ hints the interference speaker|
* All commands can have an additional __--output__ path to save the processed results, default output path is audio_examples/.

#### Examples
###### Processing single wav sample
| Task | Example |
|---|---|
|__speech denoising__| nhans_denoiser audio_examples/mixed.wav --neg=audio_examples/game_noise.wav| 
|__selective noise suppresion__| nhans_denoiser audio_examples/mixed.wav --pos=audio_examples/Silent.wav --neg=audio_examples/game_noise.wav |
|__speech separation__| nhans_separator audio_examples/mixed.wav --pos=audio_examples/target_speaker.wav --neg=audio_examples/noise_speaker.wav|

###### Processing multiple wav samples in folders
| Task | Example |
|---|---|
|__speech denoising__| nhans_denoiser audio_examples/mixed --neg=audio_examples/neg| 
|__selective noise suppresion__| nhans_denoiser audio_examples/mixed --pos=audio_examples/pos --neg=audio_examples/neg |
|__speech separation__| nhans_separator audio_examples/mixed --pos=audio_examples/target --neg=audio_examples/interference|

### Train your own N-HANS
You can download the respository to train your own selective noise suppression system and speech separation system using N-HANS architecture. 
1. To train a selective noise suppression system, please direct to N-HANS/N_HANS___Selective_Noise/ and create clean speech and noise list using __create_seeds(speech_dir, noise_dir)__, which will generate for each a .pkl file.
To train a speech separation system, please direct to N-HANS/N_HANS___Speech_Separation/ and create target and interference speech list using __create_seeds(target_dir, interference_dir)__, which will generate for each a .pkl file.
2. Run main.py script with your specifications, as FLAGS appear in the following table (default specifications were used to achieve our trained_models). The reader.py provides the training, validataion and test data pipeline and feeds the data to main.py, in which N-HANS neural networks are located. The training process will then start, its information will be shown.

| FLAGS | Default | Funcationalities |
|---|---:|---|
| --speech_wav_dir | './speech_wav_dir/' | the directory contains all speech .wav files|
| --noise_wav_dir | './noise_wav_dir/' | the directory contains all noise .wav files|
| --wav_dump_folder | './wav_dump/' | the directory to save denoised signals|
| --eval_seeds| 'valid' | evaluation is applied for 'valid' dataset. In test, change it to 'test' |
| --window_frames | 35 | number of frames of input noisy signal |
| --context_frames | 200 | number of frames of reference context signal |
| --random_slices | 50 | number of random samples from each pair of clean speech and noise signal |
| --model_name | 'nhans' | model name |
| --restore_path | '' | the path to restore trained model |
| --alg | 'sgd' | optimiser used to train N-HANS |
| --train_mb| 64 | mini-batch size for training data |
| --eval_mb| 64 | mini-batch size for validation or test data |
| --lr | 0.1 | learning rate |
| --mom | 0.0 | monentum for optimiser |
| --bn_decay| 0.95 | batch normalisation decay |
| --eval_before_training | False | Training phase: False, Test phase: True |
| --eval_after_training | True | Training phase: True, Test phase: False |
| --train_monitor_every | 1000 | show training information for each "train_monitor_every" batches |
| --eval_every | 5000 | show evaluation information for each "eval_every" training batches |
| --checkpoint_dir | './checkpoints' | directory to save checkpoints|
| --summaries | './summaries' | directory for summairies|
| --dump_results | './dump' | directory for intermediate output of model during training|

3. To test your model, __restore_path__ is set to the trained models, __--eval_seeds=test__ is also required.


## Authors and Contact information
* Shuo Liu (shuo.liu@informatik.uni-augsburg.de)
* Gil Keren
* Björn Schuller
