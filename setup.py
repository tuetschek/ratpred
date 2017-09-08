from setuptools import setup
from setuptools import find_packages


setup(
    name='ratpred',
    version='0.1.0',
    description='Referenceless quality estimation for natural language generation -- predicting human ratings',
    author='Ondrej Dusek',
    author_email='o.dusek@hw.ac.uk',
    url='https://github.com/tuetschek/ratpred',
    download_url='https://github.com/tuetschek/ratpred.git',
    license='Apache 2.0',
    install_requires=['tgen==0.2.0'],
    dependency_links=['https://github.com/UFAL-DSG/tgen/tarball/master#egg=tgen-0.2.0'],
    packages=find_packages()
)

