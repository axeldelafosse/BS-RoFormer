from setuptools import setup, find_packages

setup(
  name = 'Lossless-BS-RoFormer',
  packages = find_packages(exclude=[]),
  package_data={'bs_roformer': ['*.yaml']},
  version = '1.0.0',
  license='MIT',
  description = 'Lossless BS-RoFormer - Band-Split Rotary Transformer for SOTA Music Source Separation',
  author = 'Axel Delafosse',
  author_email = 'axeldelafosse@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/axeldelafosse/BS-RoFormer',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'music source separation'
  ],
  install_requires=[
    'beartype',
    'einops>=0.6.1',
    'librosa',
    'rotary-embedding-torch>=0.3.6',
    'torch>=2.0',
    'packaging',
    'tqdm',
    'soundfile',
    'pyyaml',
    'ml_collections',
    'requests',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
