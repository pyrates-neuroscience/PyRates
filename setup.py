from setuptools import setup, find_packages

setup(name='NMMs',
      version='0.1',
      description='Neural Mass Models',
      author='Richard Gast, Daniel Rose, Konstantin Weise',
      author_email='rgast@cbs.mpg.de',
      license='GPL3',
      packages=['NMMs',
                'NMMs/base',
                'NMMs/tests'],
      zip_safe=False)