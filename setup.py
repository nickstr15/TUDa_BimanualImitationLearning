from codecs import open
from os import path

from setuptools import setup

folder = path.abspath(path.dirname(__file__))
requires_list = []
with open(path.join(folder, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))

setup(
    name='bil',
    version='0.0.1',
    author='Nick Striebel',
    author_email='nick.striebel@stud.tu-darmstadt.de',
    packages=['robomimic_ext', 'robosuite_ext', 'utils'],
    install_requires=requires_list,
)

