########################################################################################################################
#                                          N-HANS speech separator: create_seeds                                       #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#   Description:      Creating speech and noise seeds list                            .                                #
#   Authors:          Shuo Liu, Gil Keren, Bjoern Schuller                                                             #
#   Affiliation:      Chair of Embedded Intelligence for Health Care and Wellbeing, University of Augsburg (UAU)  #
#   Version:          1.5                                                                                              #
#   Last Update:      May. 06, 2020                                                                                    #
#   Dependence Files: xxx                                                                                              #
#   Contact:          shuo.liu@informatik.uni-augburg.de                                                               #
########################################################################################################################

import tensorflow as tf
import pickle as p
import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('speech_wav_dir', './speech_wav_dir/', '')

"""
Create Pickle List for Training, Evaluation and Test
"""


def create_seeds():
    """
    Create 3 pickle files: train.pkl, valid.pkl and test.pkl for speech audio files
    :param speech_wav_dir: the directory contains three subfolders, train, valid and test, each includes audio .wav files
    """
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


if __name__ == '__main__':
    create_seeds()
