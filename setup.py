from setuptools import find_packages, setup

setup(name='marl_model',
      version='1.0',
      package_dir={'': 'src'},
      packages=find_packages("src"),
      install_requires=['tensorflow', 'plotly', 'pandas'],
      entry_points={
          "console_scripts": ["marl = marl_model.__main__:main"],
      })
