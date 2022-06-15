# Parametric Tuning of Extended Reverberation Algorithm Using Neural Network

**Master Thesis by Søren Vøgg K. Lyster**

*Sound and Music Computing, 2022-05*

## Feedback Delay Network Reverberator 



A custom implementation of a feedback delay network reverberator has been made as a VST3 using JUCE. 

<img src="images/guifade.png" alt="drawing" width="40%"/>

The source code to the repository can be found [here](https://github.com/VoggLyster/Reverberator), and the latest version compiled for x86_64 Ubuntu 18.04 can be found [here](https://github.com/VoggLyster/Reverberator/releases/latest) ![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/VoggLyster/Reverberator)

--- 
## Neural Network
The neural network for estimating the reverberator parameters can be found [here](https://github.com/VoggLyster/Reverberator). The Jupyter Notebook [ReverberatorEstimator.ipynb](https://github.com/VoggLyster/ReverberatorEstimator/blob/main/ReverberatorEstimator.ipynb) contains the setup that has been used to create the network. 

---
## Evaluation
The evaluation has been done with three different blackbox reverberators. Impulse responses have been created for all three cases with the following input audio:

<audio controls><source src='audio/input_audio.wav'></audio>

### Hall Of Fame 2 - By TC Electronic

Impulse response (Target / Estimated):

<audio controls><source src='./audio/hofTarget.wav'></audio>
<audio controls><source src='./audio/hofOutput.wav'></audio>

Drums (Blackbox / Estimated):

<audio controls><source src='./audio/HoFDrums.wav'></audio>
<audio controls><source src='./audio/est_HoFDrums.wav'></audio>

Piano (Blackbox / Estimated):

<audio controls><source src='./audio/HoFPiano.wav'></audio>
<audio controls><source src='./audio/est_HoFPiano.wav'></audio>

Vocals (Blackbox / Estimated):

<audio controls><source src='./audio/HoFVox.wav'></audio>
<audio controls><source src='./audio/est_HoFVox.wav'></audio>

[Figures for blackbox/estimated](./images/hofImages.png)

### Ableton Live Reverb - By Ableton 

Impulse response (Target / Estimated):

<audio controls><source src='./audio/abletonTarget.wav'></audio>
<audio controls><source src='./audio/abletonOutput.wav'></audio>

Drums (Blackbox / Estimated):

<audio controls><source src='./audio/AbletonDrums.wav'></audio>
<audio controls><source src='./audio/est_AbletonDrums.wav'></audio>

Piano (Blackbox / Estimated):

<audio controls><source src='./audio/AbletonPiano.wav'></audio>
<audio controls><source src='./audio/est_AbletonPiano.wav'></audio>

Vocals (Blackbox / Estimated):

<audio controls><source src='./audio/AbletonVox.wav'></audio>
<audio controls><source src='./audio/est_AbletonVox.wav'></audio>

[Figures for blackbox/estimated](./images/abletonImages.png)

### Valhalla Room - By ValhallaDSP

Impulse response (Target / Estimated):

<audio controls><source src='./audio/valTarget.wav'></audio>
<audio controls><source src='./audio/valOutput.wav'></audio>


Drums (Blackbox / Estimated):

<audio controls><source src='./audio/ValDrums.wav'></audio>
<audio controls><source src='./audio/est_ValhallaDrums.wav'></audio>

Piano (Blackbox / Estimated):

<audio controls><source src='./audio/ValPiano.wav'></audio>
<audio controls><source src='./audio/est_ValhallaPiano.wav'></audio>

Vocals (Blackbox / Estimated):

<audio controls><source src='./audio/ValVox.wav'></audio>
<audio controls><source src='./audio/est_ValhallaVox.wav'></audio>

[Figures for blackbox/estimated](./images/valImages.png)
