[![PyPI](https://img.shields.io/badge/Pypi%20Package-2020.6.22-yellow)]()
[![PyPI](https://img.shields.io/badge/Python-2.7%7C3.6--3.8-blue)]()
[![PyPI](https://img.shields.io/badge/Tensorflow-1.14%7C2.0--2.2-orange)]()
[![GitHub license](https://img.shields.io/badge/pysox-1.3.7-brightgreen)]()
[![GitHub license](https://img.shields.io/badge/scipy-1.0.1-brightgreen)]()
[![GitHub license](https://img.shields.io/badge/License-GPL%20v3-blue)]()
<!--(https://raw.githubusercontent.com/rabitt/pysox/master/LICENSE.md)-->
![M](/docs/logo_m_2.png)
<!--Latest News:  (June. 17, 2020) N-HANS is now compatible with Python3 &  Tensorflow2 --> 

__N-HANS__ is a Python toolkit for in-the-wild speech enhancement, including speech, music, and general audio denoising, separation, and selective noise or source suppression. The functionalities are realised based on two neural network models sharing the same architecture, but trained separately. The models are comprised of stacks of residual blocks, each conditioned on additional speech or environmental noise recordings for adapting to different unseen speakers or environments in real life. 

<!--In addition to a Python API, a command line interface is provided to researchers and developers:-->
<!--# Installation-->
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; __pip3 install N-HANS__

<!--__(c) 2020 Shuo Liu, Gil Keren, Björn Schuller: University of Augsburg__ published under GPL v3, see the LICENSE file for details.-->

Please direct any questions or requests to Shuo Liu (shuo.liu@informatik.uni-augsburg.de).

# Citation
If you use N-HANS or any code from N-HANS in your research work, you are kindly asked to acknowledge the use of N-HANS in your publications.

https://arxiv.org/pdf/1911.07062.pdf


# Prerequisites
* Python __3__ / Python __2.7__
### Python Dependencies
* numpy >=1.14.5
* scipy >=1.0.1
* tensorflow/tensorflow-gpu >=1.14.0 or tensorflow >= 2.0


# Usage
## Loading Models
After __pip3 install N-HANS__, users are expexted to create a N-HANS folder for conducting audio denoising or separation tasks. For linux users, commands __load_denoiser__ or __load_separator__ will assist in downloading pretrained denoising and separation models, accompanied by some audio examples. The trained models and audio examples can also be found in the above N_HANS_Selective_Noise and N_HANS_Source_Separation folders, which provides users working on other operation systems the opportunity to apply N-HANS.

## Applying N-HANS
N-HANS has been developed to process standard .wav audios with sample rate of 16kHz and coded in 16-bit Signed Integer PCM. With the embedded format converter written based on sox package, audio files of other formats are automatically to convert to this standard setting.

### Commands
| Task | Command | Discription |
|---|---|---|
|__speech denoising__| __nhans_denoiser__ __--input__ noisy.wav __--output__ denoised.wav &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; __--neg__ noise.wav | __--neg__ the environmental noise |  
|__selective noise suppresion__| __nhans_denoiser__ __--input__ noisy.wav __--output__ denoised.wav &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; __--pos__ preserve.wav __--neg__ suppress.wav | __--pos__ indicates the noise to be preserved <br> __--neg__ hints the noise to be suppressed|
|__speech separation__| __nhans_separator__ __--input__ mixed.wav __--output__ separated.wav<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; __--pos__ target.wav  __--neg__ interference.wav  | __--pos__ indicates the target speaker <br> __--neg__ hints the interference speaker|

### Examples
#### Processing single wav sample
| Task | Example |
|---|---|
|__speech denoising__| nhans_denoiser --input audio_examples/exp2_noisy.wav --output denoised.wav --neg audio_examples/exp2_noise.wav | 
|__selective noise suppresion__| nhans_denoiser --input audio_examples/exp1_noisy.wav --output denoised.wav --pos audio_examples/exp1_posnoise.wav --neg audio_examples/exp2_negnoise.wav|
|__speech separation__| nhans_separator --input audio_examples/mixed.wav --output separated.wav --pos audio_examples/target_speaker.wav --neg audio_examples/noise_speaker.wav|

#### Processing multiple wav samples in folders
Please create folders containing noisy, (positive) negative recordings, the recordings for each example in different folders should have identical filename.   

| Task | Example |
|---|---|
|__speech denoising__| nhans_denoiser --input audio_examples/noisy_dir --output denoised_dir --neg audio_examples/neg_dir| 
|__selective noise suppresion__| nhans_denoiser --input audio_examples/noisy_dir --output denoised_dir --pos audio_examples/pos_dir --neg=audio_examples/neg_dir |
|__speech separation__| nhans_separator --input audio_examples/mixed_dir --output separated_dir --pos=audio_examples/target_dir --neg=audio_examples/interference_dir|

## Train your own N-HANS
You can train your own selective audio suppression system and separation system using N-HANS architecture based on this respository. 
1. To train a selective audio suppression system, please go into N-HANS/N_HANS___Selective_Noise/ and create lists for clean speech samples and environment noises. Feed the paths of the folders that individually consists of speech .wav files and noise .wav files in __create_seeds__, which will generate two pickle files (.pkl) containing speech and noise wav files, separately. To maximally train a system that is consistent with our trained model, we provide the seed lists for the data split of AudioSet Corpus (https://research.google.com/audioset/) in our publication. To download [__AudioSet_seeds__](https://dl.dropboxusercontent.com/s/690nfzq21fb1x3u/AudioSet_Seeds.zip?dl=0).

   To train an speech separation system, please go into N-HANS/N_HANS___Speech_Separation/ and create a speech list using __create_seeds__ direct to your folder containing speech .wav files, which will produce a .pkl file.

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

3. To test your model, __restore_path__ is set to the trained models, __--eval_seeds=test__ is required.


# Authors and Contact Information
* Shuo Liu (shuo.liu@informatik.uni-augsburg.de)
* Gil Keren
* Björn Schuller
