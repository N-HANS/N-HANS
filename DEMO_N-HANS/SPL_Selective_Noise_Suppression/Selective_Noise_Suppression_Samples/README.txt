This folder corresponds to our submitted paper "Single-Channel Selective Noise Suppression: Suppressing Only Unwanted Noise while Retaining Naturalness" for Signal Processing Letters. The audio files in this folder are generated based on Librispeech [1] and AudioSet [2] test dataset.

Each example contains 5 audio files, the filename is formatted as follows
	exampleID_SpeechID_postiveNoiseID_negativeNoiseID_positiveSNR_negativeSNR_suffix.wav

SpeechID: 	 the filename of speech segment in Librespeech.
positiveNoiseID: the filename of audio sample in Audio Set which serves as the noise to be preserved.  
negativeNoiseID: the filename of audio sample in Audio Set which serves as the noise to be suppressed. 
positiveSNR: 	 the SNR for positive noise, labeled in dB.
negativeSNR: 	 the SNR for negative noise, labeled in dB.
suffix:          "mixed" indicates the noisy audio comprised of speech signal and positive and negative noise,
	         "posNoise" indicates the noise to be preserved,
	         "negNoise" indicates the noise to be suppressed,
	         "target" indicates the ideal output, which should be the composition of speech and postive noise,
	         "denoised" indicates the output of our selective noise suppression system, which should be close to "target"

Our proposed model processed the "..._mixed.wav" and produced the "..._denoised.wav", which is expected to be as close as "..._target.wav". 

[1] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “Librispeech: an ASR corpusbased on public domain audio books,” in Proc. ICASSP, Brisbane, Australia, 2015, pp.5206–5210.
[2] J. Gemmeke, D. Ellis, D. Freedman, A. Jansen, W. Lawrence, R. Moore, M. Plakal, andM. Ritter, “Audio Set: An ontology and human-labeled dataset for audio events,” in Proc. ICASSP, New Orleans, LA, USA, 2017, pp. 776–780.
