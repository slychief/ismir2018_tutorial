# Deep Learning for Music Classification using Keras

(c) 2018 by



http://ismir2018.ircam.fr/pages/events-tutorial-04.html





|                                                              |                                                              |                                                              |
|:------------------------------------------------------------:| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="http://ismir2018.ircam.fr/images/tutorial_photo_schindler.jpg" width="200px"> | **Alexander Schindler** is member of the Music Information Retrieval group at the Technical University since 2010 where he actively participates in research, various international projects and currently finishes his Ph.D on audio-visual analysis of music videos. He participates in teaching MIR, machine learning and DataScience. Alexander is currently employed as scientist at the AIT Austrian Institute of Technology where he is responsible for establishing a deep learning group. In various projects he focusses on deep-learning based audio-classification, audio event-detection and audio-similiarity retrieval tasks. | [Website](http://ifs.tuwien.ac.at/~schindler), [Twitter](https://twitter.com/Slychief) |
| <img src="http://ismir2018.ircam.fr/images/tutorial_photo_lidy.png" width="100"> | **Thomas Lidy** has been a researcher in music information retrieval in combination with machine learning at TU Wien since 2004. Since 2015, he has been focusing on how Deep Learning can further improve music & audio analysis, winning 3 international benchmarking contests. He is currently the Head of Machine Learning at Musimap, a company that uses Deep Learning to analyze styles, moods and emotions in the global music catalog, in order to create emotion-aware search & recommender engines that empower music supervisors to find the music for their needs and music streaming platforms to deliver the perfect playlists according to people's mood. | [Website]( https://www.linkedin.com/in/thomaslidy/),[Twitter](https://twitter.com/LidyTom) |
| <img src="http://ismir2018.ircam.fr/images/tutorial_photo_boeck.jpg" width="100"> | **Sebastian Böck** received his diploma degree in electrical engineering from the Technical University in Munich in 2010 and his PhD in computer science from the Johannes Kepler University Linz. He continued his research at the Austrian Research Institute for Artificial Intelligence (OFAI) and recently also joined the MIR team at the Technical University of Vienna. His main research topic is the analysis of time event series in music signals, with a strong focus on artificial neural networks. | Website,Twitter                                              |



also visit: https://www.meetup.com/Vienna-Deep-Learning-Meetup

# Abstract

Deep Learning has become state of the art in visual computing and continuously emerges into the Music Information Retrieval (MIR) and audio retrieval domain. To bring attention to this topic we provide an introductory tutorial on deep learning for MIR. Besides a general introduction to neural networks, the tutorial covers a wide range of MIR relevant deep learning approaches. Convolutional Neural Networks are currently a de-facto standard for deep learning based audio retrieval. Recurrent Neural Networks have proven to be effective in onset detection tasks such as beat or audio-event detection. Siamese Networks have shown to be effective in learning audio representations and distance functions specific for music similarity retrieval. We introduce these different neural network layer types and architectures on the basis of standard MIR tasks such as music classification, similarity estimation and onset detection. We will incorporate both academic and industrial points of view into the tutorial. The tutorial will be accompanied by a Github repository for the presented content as well as references to state of the art work and literature for further reading. This repository will remain public after the conference.

# Tutorial Outline

**Part 1 - Deep Learning Basics** ([Slides](https://github.com/slychief/mlprague2018_tutorial/blob/master/Tutorial%20Slides/Part%201%20-%20Basics%20of%20digital%20audio%20and%20deep%20learning.pdf))
  * Audio Processing Basics ([Jupyter Notebook](https://github.com/slychief/mlprague2018_tutorial/blob/master/Part%201a%20-%20Audio%20Basics.ipynb))
  * History of Neural Networks
  * What is Deep Learning
  * Neural Network Concepts
  * Coding Examples ([Jupyter Notebook](https://github.com/slychief/mlprague2018_tutorial/blob/master/Part%201b%20-%20Deep%20Learning%20Basics.ipynb))

**Part 2 - Convolutional Neural Networks** ([Slides](https://github.com/slychief/mlprague2018_tutorial/blob/master/Tutorial%20Slides/Part%202%20-%20Introduction%20to%20CNNs.pdf))
  * Difference CNN – RNN
  * How CNNs work (Layers, Filters, Pooling)
  * Application Domains and how to use in Music
  * Coding Examples ([Jupyter Notebook](https://github.com/slychief/mlprague2018_tutorial/blob/master/Part%202%20-%20Music%20speech%20classification.ipynb))

**Part 3 - Instrumental, Genre and Mood Analysis** ([Slides](https://github.com/slychief/mlprague2018_tutorial/blob/master/Tutorial%20Slides/Part%203%20-%20Music%20AI%20at%20Musimap.pdf))
  * Large-scale Music AI at Musimap
  * Instrumental vs. Vocal Detection
  * Genre Recognition
  * Mood Detection
  * Coding Examples ([Jupyter Notebook](https://github.com/slychief/mlprague2018_tutorial/blob/master/Part%203%20-%20Instrumental%20-%20Genre%20-%20Mood%20detection.ipynb))

**Part 4 - Advanced Deep Learning**
  * Similarity Retrieval ([Jupyter Notebook](https://github.com/slychief/mlprague2018_tutorial/blob/master/Part%204a%20-%20Distance%20Based%20Search.ipynb))
  * Siamese Networks ([Jupyter Notebook](https://github.com/slychief/mlprague2018_tutorial/blob/master/Part%204b%20-%20Siamese%20Networks.ipynb))
  * Learning Audio Representation from Tag Similarity ([Jupyter Notebook](https://github.com/slychief/mlprague2018_tutorial/blob/master/Part%204c%20-%20Siamese%20Networks%20with%20Tag%20Similarity.ipynb))

# Tutorial Requirements

For the tutorials, we use iPython / Jupyter notebook, which allows to program and execute Python code interactively in the browser.

### Viewing Only

If you do not want to install anything, you can simply view the tutorials' content in your browser, by clicking on the tutorial's filenames listed above in the GIT file listing.

The tutorials will open in your browser for viewing.

### Interactive Coding

If you want to follow the tutorials by actually executing the code on your computer, please [install first the pre-requisites](#installation-of-pre-requisites) as described below.

After that, to run the tutorials go into the `mlprague2018_tutorial` folder and start from the command line:

`jupyter notebook`


# Installation of Pre-requisites

## Install Python 3.x

Note: On most Mac and Linux systems Python is already pre-installed. Check with `python --version` on the command line whether you have Python 3.x installed.

Otherwise install Python 3.5 from https://www.python.org/downloads/release/python-350/

## Install Python libraries:

### Mac, Linux or Windows

(on Windows leave out `sudo`)

Important note: If you have Python 2.x and 3.x installed in parallel, replace `pip` by `pip3` in the following commands:

```
sudo pip install --upgrade jupyter
```

Try if you can open 
```
jupyter notebook
```
on the command line. 

Then download or clone the Tutorials from this GIT repository:

```
git clone https://github.com/slychief/mlprague2018_tutorial.git
```
or download https://github.com/slychief/mlprague2018_tutorial/archive/master.zip <br/>
unzip it and rename the folder to `mlprague2018_tutorial`.

Install the remaining Python libraries needed:

Either by:

```
sudo pip install Keras>=2.1.1 tensorflow scikit-learn>=0.18 pandas librosa spotipy matplotlib
```

or, if you downloaded or cloned this repository, by:

```
cd mlprague2018_tutorial
sudo pip install -r requirements.txt
```

### Optional for GPU computation

If you want to train your neural networks on your GPU (which is faster, but not necessarily needed for this tutorial),
you have to install the specific GPU version of Tensorflow:

```
sudo pip install tensorflow-gpu
```

and also install the following:

* [NVidia drivers](http://www.nvidia.com/Download/index.aspx?lang=en-us)
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [cuDNN](https://developer.nvidia.com/cudnn) (requires registration with Nvidia)


## Install Audio Decoder

In order to decode MP3 files (used in the MagnaTagAtune data set) you will need to install FFMpeg on your system.

* Linux: `sudo apt-get install ffmpeg`
* Mac: download FFMPeg for Mac: http://ffmpegmac.net and make sure ffmpeg is on PATH
* Windows: download https://github.com/tuwien-musicir/rp_extract/blob/master/bin/external/win/ffmpeg.exe and make sure it is on the PATH


## Download Prepared Datasets

Please download the following data sets for this tutorial:

**MagnaTagAtune**

Prepared Features and Metadata: https://owncloud.tuwien.ac.at/index.php/s/3RPE4UutTaXXAib (212MB)

Audio-files: https://owncloud.tuwien.ac.at/index.php/s/bxY87m3k4oMaoFl (702MB)

These are prepared versions from the original datasets described below.


# Credits

The following helper Python libraries are used in these tutorials:

* `audiofile_read.py` and `audio_spectrogram.py`: by Thomas Lidy and Alexander Schindler, taken from the [RP_extract](https://github.com/tuwien-musicir/rp_extract) git repository
* `wavio.py`: by Warren Weckesser

The data sets we use in the tutorials are from the following sources:

* MagnaTagAtune: http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset

(don't download them from there but use the prepared datasets from the two owncloud links above)
