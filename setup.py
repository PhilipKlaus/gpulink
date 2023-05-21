import codecs
import os
import sys

from setuptools import setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


if sys.version_info < (3, 7):
    sys.exit('Sorry, Python < 3.7 is not supported')

with open("PYPI.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

tests_require = ['pytest', 'pytest-mock', 'pytest-cov']

setup(
    name="gpulink",
    version=get_version("gpulink/__init__.py"),
    author="Philip Klaus",
    description="A simple tool for monitoring and displaying GPU stats",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PhilipKlaus/gpu-link",
    project_urls={
        "Bug Tracker": "https://github.com/PhilipKlaus/gpu-link/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=['gpulink', 'gpulink.cli', 'gpulink.devices', 'gpulink.plotting', 'gpulink.recording',
              'gpulink.threading'],
    python_requires=">=3.7",
    install_requires=[
        "pynvml == 11.5.0",
        "matplotlib >= 3.7.1",
        "numpy >= 1.24.3",
        "tabulate >= 0.9.0",
        "colorama >= 0.4.6",
        "click >= 8.1.3"
    ],
    tests_require=tests_require,
    extras_require={"test": tests_require},
    entry_points={
        'console_scripts': ['gpulink=gpulink.__main__:main'],
    },

)
