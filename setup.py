
from distutils.core import setup

setup(name='recursive-marl-model',
      version='1.0',
      package_dir={'': 'src'},
      packages=['recursive_marl_model'],
      requires=['tensorflow', 'plotly'],
     )
