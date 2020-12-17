from setuptools import setup, find_packages
import streamer

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pyclay-streamer',
    version=streamer.__version__,
    description='OpenCV Video Streaming and Recording API',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cm107/streamer",
    author='Clayton Mork',
    author_email='mork.clayton3@gmail.com',
    license='MIT License',
    packages=find_packages(
        where='.',
        include=['streamer*']
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'opencv-python>=4.1.1.26',
        'numpy>=1.17.2',
        'pylint>=2.4.2',
        'pyclay-common_utils>=0.2.5'
    ],
    python_requires='>=3.7'
)
