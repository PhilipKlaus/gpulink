from setuptools import setup

with open("PYPI.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gpulink-phil.k",
    version="0.1.2",
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
    packages=['.gpulink'],
    python_requires=">=3.6",
    install_requires=[
        "pynvml >= 11.4.1",
        "matplotlib >= 3.5.1",
        "numpy >= 1.22.2"
    ],
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['gpulink=gpulink.cmd:main'],
    },

)
