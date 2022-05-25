# Parameter Estimation of Reverberation Using a Neural Network

A good sounding reverb can be a tricky audio effect to achieve. An artificial reverberation algorithm with multiple filters and delay lines can consist of a high number of adjustable parameters and the task of tweaking these parameters to achieve the desired reverberation can take hours or days even for a skilled audio engineer. Estimating a large number of parameters to reach a desired target is a use case that fits well into the subject of machine learning and neural networks. For this project I propose an adaptation of a neural network model to estimate a large set of parameters of a
reverberator with the purpose of tuning that reverberator to
emulate a target reverberated audio signal

---
## Content
This repository contains: 
* a Jupyter Notebook with the neural network setup - [ReverberatorEstimator.ipynb](https://github.com/VoggLyster/ReverberatorEstimator/blob/main/ReverberatorEstimator.ipynb)
* a folder containing python implementation of custom layers and loss - [ReverberatorEstimator](https://github.com/VoggLyster/ReverberatorEstimator/tree/main/ReverberatorEstimator)
* a small dataset for running the training - [Dataset](https://github.com/VoggLyster/ReverberatorEstimator/tree/main/Dataset)
* a small Notebook for checking and writing out the parameters of a VST3 plugin [TestParameters.ipynb](https://github.com/VoggLyster/ReverberatorEstimator/blob/main/TestParameters.ipynb)
* a requirements.txt for Python environment setup - [requirements.txt](https://github.com/VoggLyster/ReverberatorEstimator/blob/main/requirements.txt)

---
## Examples
The code can be tried out using the Jupyter Notebook [ReverberatorEstimator.ipynb](https://github.com/VoggLyster/ReverberatorEstimator/blob/main/ReverberatorEstimator.ipynb).

Audio examples can be found at https://vogglyster.github.io/ReverberatorEstimator/

---

## Feedback Delay Network Reverberator 

A custom implementation of a feedback delay network reverberator has been made as a VST3 using JUCE. 

<img src="docs/images/guifade.png" width="50%" />


The source code to the repository can be found [here](https://github.com/VoggLyster/Reverberator), and the latest version compiled for x86_64 Ubuntu 18.04 can be found [here](https://github.com/VoggLyster/Reverberator/releases/latest) ![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/VoggLyster/Reverberator)

---
