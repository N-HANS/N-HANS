<img width="800" height="193" src="/docs/logoB.png"/>


__N-HANS__ is a Python toolkit for in-the-wild speech enhancement, including speech, music, and general audio denoising, separation, and selective noise or source suppression. The functionalities are realised based on two neural network models sharing the same architecture, but trained separately. The models are comprised of stacks of residual blocks, each conditioned on additional speech or environmental noise recordings for adapting to different unseen speakers or environments in real life. 

In addition to a Python API, a command line interface is provided to researchers and developers:
                                    __pip install N-HANS__.

__(c) 2019 Shuo Liu, Gil Keren, Björn Schuller: University of Augsburg__ published under GPL v3, see the LICENSE file for details.

Please direct any questions or requests to Shuo Liu (shuo.liu@informatik.uni-augsburg.de).

# Citation
If you use N-HANS or any code from N-HANS in your research work, you are kindly asked to acknowledge the use of N-HANS in your publications.

https://arxiv.org/pdf/1911.07062.pdf


# Prerequisites
* Python2.7
### Python Dependencies
* numpy 1.14.5
* scipy 1.0.1
* six 1.10.0  
* tensorflow 1.14.0 or tensforflow-gpu 1.14.0

# Usage
## Loading Models
After __pip install N-HANS__, users are expexted to create a workspace for audio denoising or separation task, and then Linux users can utilise commands __load_denoiser__ or __load_separator__ to download the trained models and audio examples into the workspace. For other operation systems, please download the __trained_model__ and __audio_examples__ in the corresponding N_HANS subfolders, and put into the created workspace.

## Applying N-HANS
N-HANS processes standard .wav audios with sample rate of 16kHz and coded in 16-bit Signed Integer PCM. Other formats are sugguested to convert to this standard setting.
### Commands
| Task | Command | Discription |
|---|---|---|
|__speech denoising__| __nhans_denoiser__ __--input__ noisy.wav &nbsp;&nbsp;&nbsp;&nbsp; __--neg__=noise.wav | __--neg__ implicates the environmental noise |  
|__selective noise suppresion__| __nhans_denoiser__ __--input__ noisy.wav __--pos__=preserve.wav __--neg__=suppress.wav | __--pos__ implicates the noise to keep &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; __--neg__ hints the noise to suppress |
|__speech separation__| __nhans_separator__ __--mixed.wav__ __--pos__=target.wav  __--neg__=interference.wav | __--pos__ implicates the target speaker &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  __--neg__ hints the interference speaker|
* All commands can have an additional __--output__ path to save the processed results, default output path is audio_examples/.

### Examples
#### Processing single wav sample
| Task | Example |
|---|---|
|__speech denoising__| nhans_denoiser audio_examples/exp2_noisy.wav --neg=audio_examples/exp2_noise.wav| 
|__selective noise suppresion__| nhans_denoiser audio_examples/exp1_noisy.wav --pos=audio_examples/exp1_posnoise.wav --neg=audio_examples/exp2_negnoise.wav |
|__speech separation__| nhans_separator audio_examples/mixed.wav --pos=audio_examples/target_speaker.wav --neg=audio_examples/noise_speaker.wav|

#### Processing multiple wav samples in folders
Please create folders containing noisy, (positive) negative recordings, the recordings for each sample in different folders should have an identical filename.   

| Task | Example |
|---|---|
|__speech denoising__| nhans_denoiser audio_examples/noisy --neg=audio_examples/neg| 
|__selective noise suppresion__| nhans_denoiser audio_examples/noisy --pos=audio_examples/pos --neg=audio_examples/neg |
|__speech separation__| nhans_separator audio_examples/mixed --pos=audio_examples/target --neg=audio_examples/interference|

## Train your own N-HANS
You can download the respository to train your own selective audio suppression system and separation system using N-HANS architecture. 
1. To train a selective audio suppression system, please go into N-HANS/N_HANS___Selective_Noise/ and create clean speech and noise list using __create_seeds__ specific to your folders containg speech .wav files and noise .wav files, which will generate for two .pkl files. The AudioSet seeds list that we used for generating training, validation and test set in our publication is provided as .pkl files. To download [__AudioSet_seeds__](https://dl.dropboxusercontent.com/s/cmmi3c89awy8jht/AudioSet_seeds.zip?dl=0).

   To train an speech separation system, please go into N-HANS/N_HANS___Speech_Separation/ and create a speech list using __create_seeds__ specific to your folder containing speech .wav files, which will produce a .pkl file.

2. Run main.py script with your specifications indicated by FLAGS appear in the following table (default specifications were used to achieve our trained_models). The reader.py provides the training, validataion and test data pipeline and feeds the data to N-HANS neural networks constructed in main.py. 

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


# Authors and Contact Information
* Shuo Liu (shuo.liu@informatik.uni-augsburg.de)
* Gil Keren
* Björn Schuller
