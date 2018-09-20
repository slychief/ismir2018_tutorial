# Deep Learning for Music Information Retrieval

*(c) 2018 by Alexander Schindler, Thomas Lidy and Sebastian Böck*

This repository contains slides, code and further material for the "Deep Learning for MIR" tutorial held at the *19th International Society for Music Information Retrieval Conference* in Paris, France, from September 23-27, 2018.


**Tutorial Web-site:** http://ismir2018.ircam.fr/pages/events-tutorial-04.html



## Authors / Lecturers

|                                                              |                                                              |                                                              |
|:------------------------------------------------------------:| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="http://ismir2018.ircam.fr/images/tutorial_photo_schindler.jpg" width="200px"> | **Alexander Schindler** is member of the Music Information Retrieval group at the Technical University since 2010 where he actively participates in research, various international projects and currently finishes his Ph.D on audio-visual analysis of music videos. He participates in teaching MIR, machine learning and DataScience. Alexander is currently employed as scientist at the AIT Austrian Institute of Technology where he is responsible for establishing a deep learning group. In various projects he focusses on deep-learning based audio-classification, audio event-detection and audio-similiarity retrieval tasks. | [Website](http://ifs.tuwien.ac.at/~schindler), [Twitter](https://twitter.com/Slychief) |
| <img src="http://ismir2018.ircam.fr/images/tutorial_photo_lidy.png" width="100"> | **Thomas Lidy** has been a researcher in music information retrieval in combination with machine learning at TU Wien since 2004. Since 2015, he has been focusing on how Deep Learning can further improve music & audio analysis, winning 3 international benchmarking contests. He is currently the Head of Machine Learning at Musimap, a company that uses Deep Learning to analyze styles, moods and emotions in the global music catalog, in order to create emotion-aware search & recommender engines that empower music supervisors to find the music for their needs and music streaming platforms to deliver the perfect playlists according to people's mood. | [Website]( https://www.linkedin.com/in/thomaslidy/),[Twitter](https://twitter.com/LidyTom) |
| <img src="http://ismir2018.ircam.fr/images/tutorial_photo_boeck.jpg" width="100"> | **Sebastian Böck** received his diploma degree in electrical engineering from the Technical University in Munich in 2010 and his PhD in computer science from the Johannes Kepler University Linz. He continued his research at the Austrian Research Institute for Artificial Intelligence (OFAI) and recently also joined the MIR team at the Technical University of Vienna. His main research topic is the analysis of time event series in music signals, with a strong focus on artificial neural networks. | Website,Twitter                                              |



also visit: https://www.meetup.com/Vienna-Deep-Learning-Meetup

## Abstract

Deep Learning has become state of the art in visual computing and continuously emerges into the Music Information Retrieval (MIR) and audio retrieval domain. To bring attention to this topic we provide an introductory tutorial on deep learning for MIR. Besides a general introduction to neural networks, the tutorial covers a wide range of MIR relevant deep learning approaches. Convolutional Neural Networks are currently a de-facto standard for deep learning based audio retrieval. Recurrent Neural Networks have proven to be effective in onset detection tasks such as beat or audio-event detection. Siamese Networks have shown to be effective in learning audio representations and distance functions specific for music similarity retrieval. We introduce these different neural network layer types and architectures on the basis of standard MIR tasks such as music classification, similarity estimation and onset detection. We will incorporate both academic and industrial points of view into the tutorial. The tutorial will be accompanied by a Github repository for the presented content as well as references to state of the art work and literature for further reading. This repository will remain public after the conference.

## Tutorial Outline

**Part 0 - Audio Processing Basics**

* Audio Processing in Python ([Jupyter Notebook](./Part_0_Audio_Basics.ipynb))
* Preparing data and meta-data for this tutorial ([Jupyter Notebook](./Part_0_Prepare_dataset_Magnatagatune.ipynb))

**Part 1 - Audio Classification / Tagging (with CNNs)**

  * Introduction (Slides)
  * Instrumental vs. Vocal Detection ([Jupyter Notebook](./Part_1_Instrumental_Genre_Mood_detection.ipynb))
  * Convolutional Neural Networks
  * Genre Classification
  * Mood Recognition

**Part 2 - Music Similarity Retrieval (with Siamese Networks)** 

  * Distance-based search on handcrafted music features ([Jupyter Notebook](./Part_2a_Distance_Based_Search.ipynb))
  * Representation learning Siamese Neural Networks ([Jupyter Notebook](./Part_2b_Siamese_Networks.ipynb))
  * Optimizing representation learning
  * Learning music similartiy from tags ([Jupyter Notebook](Part_2c_Siamese_Networks_with_Tag_Similarity.ipynb))

**Part 3 - Onset and Beat Detection (with RNNs)**

  * Recurrent Neural Networks (Slides)



## Tutorial Requirements

For the tutorials, we use iPython / Jupyter notebook, which allows to program and execute Python code interactively in the browser.

### Viewing Only

If you do not want to install anything, you can simply view the tutorials' content in your browser, by clicking on the tutorial's filenames listed above in the GIT file listing.

The tutorials will open in your browser for viewing.

### Interactive Coding

If you want to follow the tutorials by actually executing the code on your computer, please [install first the pre-requisites](#installation-of-pre-requisites) as described below.

After that, to run the tutorials go into the `ismir2018_tutorial` folder and start from the command line:

`jupyter notebook`

### Interactive Audio Listening examples within the Browser

The browser-based Jupyter notebooks contain HTML5 audio components to directly listen to predicted results such as for the task of music similarity retrieval. Almost all recent Internet Browsers prohibit direct file access due to decent security issues. Thus, the files have to be provided via the correct protocoll.

To enable the audio samples within the browser, [download](https://owncloud.tuwien.ac.at/index.php/s/bxY87m3k4oMaoFl) and extract the audio files to a directory on your computer. Open a Python Terminal and change to the `mp3_full` directory. Then host a simple Web-server with the following command:

`python -m http.server 9999 --bind 127.0.0.1`

This will server the directory via HTTP. The supplied parameters localhost and port-number are used equally in the Jupyter notebooks.



## Download Prepared Datasets

Please download the following data sets for this tutorial:

**MagnaTagAtune**

Prepared Features and Metadata: https://owncloud.tuwien.ac.at/index.php/s/VyDlQKmsA2EFAhv (209MB)

Audio-files: https://owncloud.tuwien.ac.at/index.php/s/bxY87m3k4oMaoFl (702MB)

These are prepared versions from the original datasets described below.




## Installation of Pre-requisites

### Install Python 3.x

Note: On most Mac and Linux systems Python is already pre-installed. Check with `python --version` on the command line whether you have Python 3.x installed.

Otherwise install Python 3.5 from https://www.python.org/downloads/release/python-350/

We recommend to install the Anaconda Python Distribution due to coverage of scientific Python libraries (most of the libs required in this tutorial are already included): https://www.anaconda.com/download/



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
git clone https://github.com/slychief/ismir2018_tutorial.git
```
or download https://github.com/slychief/ismir2018_tutorial/archive/master.zip <br/>
unzip it and rename the folder to `ismir2018_tutorial`.

Install the remaining Python libraries needed:

Either by:

```
sudo pip install Keras tensorflow scikit-learn pandas numpy librosa matplotlib progressbar2 seaborn scipy
```

or, if you downloaded or cloned this repository, by:

```
cd ismir2018_tutorial
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


### Install Audio Decoder

In order to decode MP3 files (used in the MagnaTagAtune data set) you will need to install FFMpeg on your system.

* Linux: `sudo apt-get install ffmpeg`
* Mac: download FFMPeg for Mac: http://ffmpegmac.net and make sure ffmpeg is on PATH
* Windows: download https://github.com/tuwien-musicir/rp_extract/blob/master/bin/external/win/ffmpeg.exe and make sure it is on the PATH



## Further Reading / Information

* **Music Information Retrieval**
  * [Standard MIR paper by J. Steven Downie](http://www.music.mcgill.ca/~ich/classes/mumt611_06/downie_mir_arist37.pdf)
  * [ISMIR Proceedings Database](https://dblp.uni-trier.de/db/conf/ismir/index.html)
  * [List of Datasets](https://www.audiocontentanalysis.org/data-sets/) by Alexander Lerch
  * Book [Music Similarity and Retrieval](https://www.springer.com/de/book/9783662497203) by Peter Knees and Markus Schedl
* **Music/Audio Content Processing**
  * [Librosa Tutorial](https://librosa.github.io/librosa/tutorial.html) - Music/Audio processing in Python
  * Book: [An Introduction to Audio Content Analysis](https://ieeexplore.ieee.org/xpl/bkabstractplus.jsp?bkn=6266785) by  Alexander Lerch
* **Deep Learning for Music**
  * [Deep Learning for Music (DL4M)](https://github.com/ybayle/awesome-deep-learning-music) - extensive collection of papers and Github repositories!
  * [dl4mir: A Tutorial on Deep Learning for MIR](https://github.com/keunwoochoi/dl4mir) by [Keunwoo Choi](https://keunwoochoi.wordpress.com) 
* **Deep Learning in General**
  * [Keras](https://keras.io/): Python Deep Learning library
  * [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/) - Stanford course on deep learning
  * [Vienna Deep Learning Meetup](https://github.com/vdlm/meetups)



## Credits

The following helper Python libraries are used in these tutorials:

* The [RP_extract](https://github.com/tuwien-musicir/rp_extract) feature extractor and content descriptors by Thomas Lidy and Alexander Schindler

The data sets we use in the tutorials are from the following sources:

* MagnaTagAtune: http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset

(don't download them from there but use the prepared datasets from the two owncloud links above)
