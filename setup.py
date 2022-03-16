from setuptools import setup

setup(
    name="gpulink",
    version="0.1.0",
    author="Philip Klaus",
    description="A simple tool for monitoring and displaying GPU stats",
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
    entry_points={
        'console_scripts': ['gpulink=gpulink.cmd:main'],
    },

)
