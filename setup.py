from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='streamer',
    version='0.1',
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
        'common_utils @ https://github.com/cm107/common_utils/archive/v0.1.zip#egg=common_utils-0.1',
        'logger @ https://github.com/cm107/logger/archive/v0.1.zip#egg=logger-0.1'
    ],
    python_requires='>=3.7'
)
