import codecs
import os
import sys

try:
	from setuptools import setup
except:
	from distutils.core import setup



def read(fname):
	return codecs.open(os.path.join(os.path.dirname(__file__), fname)).read()


platforms = ['linux']
classifiers = [
    'Development Status :: 3 - Alpha',
    'Topic :: Multimedia :: Sound/Audio',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
]

install_requires = [
    'numpy==1.14.5',
    'tensorflow_gpu==1.15.0',
    'six==1.10.0',
    'scipy==1.0.1'
]

    
setup(name='N-HANS',
      version='1.6',
      description='Neuro-Holistic Audio-eNhancement System',
      long_description='',
	  packages=['N_HANS','N_HANS.N_HANS___Selective_Noise','N_HANS.N_HANS___Source_Separation'],
      author="Shuo Liu, Gil Keren, Bjoern Schuller",
      author_email="shuo.liu@informatik.uni-augsburg.de",
      url="https://github.com/N-HANS/N-HANS",
      license="License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
      platforms=['linux'],
      classifiers=classifiers,
      install_requires=install_requires,
	  scripts = [],
	  entry_points = {
		'console_scripts': [
			'nhans_denoiser = N_HANS.N_HANS___Selective_Noise.apply:main',
			'load_denoiser = N_HANS.N_HANS___Selective_Noise.load_model:main',
			'nhans_separator = N_HANS.N_HANS___Source_Separation.apply:main',
			'load_separator = N_HANS.N_HANS___Source_Separation.load_model:main',
		]
	  }
)
