from setuptools import setup, find_packages

# fixme: folder reference needs to be fixed

setup(name='BrainNetworks',
      version='0.1',
      description='Neural Mass Modeling Framework',
      author='Richard Gast, Daniel Rose, Konstantin Weise',
      author_email='rgast@cbs.mpg.de',
      license='GPL3',
      packages=['core/axon',
                'core/synapse',
                'core/population',
                'core/network',
                'tests'],
      zip_safe=False,
      python_requires='>=2.7',
      install_requires=['numpy', 'matplotlib', 'scipy'])
