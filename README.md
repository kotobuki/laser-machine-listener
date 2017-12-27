# Laser Machine Listener

This is a project to embody a machine that recognises the operating state by listening to the sound emitted by a laser machine during processing. We adopt deep learning which is one of machine learning for the method, [Raspberry Pi](https://www.raspberrypi.org/) of the microcomputer board widely used around the world for the platform.

これは、レーザー加工機が加工中に発する音を聞いて動作状態を識別する機械を具現化するプロジェクトです。手法には機械学習の一つであるディープラーニングを、プラットフォームには世界中で広く活用されているマイコンボード[Raspberry Pi](https://www.raspberrypi.org/)を採用しています。

[![Laser Machine Sound Classification Test](http://img.youtube.com/vi/HSIRi2tW8kI/0.jpg)](https://youtu.be/HSIRi2tW8kI)

## Problem

One of the typical digital machine tools installed in fab facilities is a laser machine that cuts the material or marks it on the surface with a laser beam. It can be used relatively quickly since it can be processed in a short time based on 2-dimensional data, and it is a machine tool which is quite deep inside which you can use advanced machining if you can master it.

One of the mistakes that a beginner is likely to cause when working with a laser machine is a shift in focus. For example, when processing a warped wood or a thin paper, it happens when you focus on a part and come off at another place. Also, it happens when you inadvertently forget to set up in a scenario where multiple users alternate in a short time.

When used in a state where the focus is not precisely matched, not only it can not be cut as initially intended but also more smoke than usual. As a result, even when used for a short time, many dirt adheres to mirrors and lenses of the laser machine, clogging filter of the dust collector, etc. occurs.

ファブ施設に設置されているデジタル工作機械の中で代表的なものの一つに、レーザー光線で材料を切断したり、表面にマーキングしたりするレーザー加工機があります。2次元のデータを元に短時間で加工できることから比較的手軽に使い始めることができ、使いこなせば高度な加工までできる、かなり奥の深い工作機械です。

初心者がレーザー加工機を使って加工する際に起きやすいミスの1つに、焦点のズレがあります。例えば、反りのある木材や薄い紙を加工する際、一部で焦点を合わせても別のところで外れてしまう時に発生します。また、複数の利用者が短時間に交代で使用する場面において、うっかり設定し忘れてしまったときにも起きます。

焦点が正確に合っていない状態で使用すると、本来意図したように切断できないだけでなく、通常よりも多くの煙が発生します。このため、短時間の使用でもレーザー加工機のミラーやレンズに多くの汚れが付着する、集塵機のフィルターが詰まる、といったことが起きてしまいます。

[![Operating sounds during processing with a laser machine](http://img.youtube.com/vi/5layASwZVbk/0.jpg)](https://youtu.be/5layASwZVbk)

## Idea

According to an experienced staff, it is possible to judge to some extent the shift of the focus and the condition of the filter just by listening to the operating sound of the laser machine. Therefore, I challenged to estimate the state of a laser machine using a Raspberry Pi and machine learning.

熟練したスタッフによると、レーザー加工機の動作音を聞くだけで焦点のズレやフィルターの状態をある程度判断できるそうです。そこで、Raspberry Piと機械学習を用いてレーザー加工機の状態を推定することにチャレンジしました。

## Applications

- Call attention to a user when detecting a case where it seems that it is not used correctly, such as a shift in focus.
- Record the occupancy rate during a certain period, and what kind of processing and how long it is done to let an administrator can analyse later on.

- 焦点がズレているなど正しく使用できていないと思われる場合を検出したら使用者に対して注意を促す。
- ある期間中の稼働率や、どのような加工をどの程度の時間行なっているかを管理者が後で分析できるように記録する。

## Prerequisites

### Software

- [Keras](https://keras.io/) with [TensorFlow](https://www.tensorflow.org/), or [Chainer](https://chainer.org/)
- [librosa](https://librosa.github.io/)
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/)
- Python 3
- [Jupyter Notebook](http://jupyter.org/)

#### Install on Raspbian

```bash
$ sudo apt-get install python3-pyaudio
$ sudo pip3 install resampy==0.1.5 librosa
$ sudo apt-get install libblas-dev liblapack-dev python-dev libatlas-base-dev gfortran python-setuptools
```

```bash
$ curl -O http://ci.tensorflow.org/view/Nightly/job/nightly-pi-python3/lastSuccessfulBuild/artifact/output-artifacts/tensorflow-1.4.0-cp34-none-any.whl
$ mv tensorflow-1.4.0-cp34-none-any.whl tensorflow-1.4.0-cp35-none-any.whl
$ sudo pip3 install tensorflow-1.4.0-cp35-none-any.whl
$ sudo apt-get install python3-h5py
$ sudo pip3 install keras
```

```bash
$ sudo pip3 install chainer
```

### Hardware

- Raspberry Pi [3 Model B](https://www.raspberrypi.org/products/raspberry-pi-3-model-b/) or [Zero W](https://www.raspberrypi.org/products/raspberry-pi-zero-w/): 1
- Micro SD card (more than 8GB): 1
- Micro USB cable: 1
- USB power adapter: 1
- [ReSpeaker 2-Mics Pi HAT](https://www.seeedstudio.com/ReSpeaker-2-Mics-Pi-HAT-p-2874.html): 1

## Contributions

- Implementation by Shigeru Kobayashi
- Preparation of the data for the laser machine and operation by Chisato Takami

## References

- Saeed, Aaqib. "Urban Sound Classification, Part 1." Aaqib Saeed. September 3, 2016. Accessed November 25, 2017. [http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/](http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/).
- Shaikh, Faizan. "Getting Started with Audio Data Analysis (Voice) using Deep Learning." Analytics Vidhya. August 23, 2017. Accessed November 25, 2017. [https://www.analyticsvidhya.com/blog/2017/08/audio-voice-processing-deep-learning/](https://www.analyticsvidhya.com/blog/2017/08/audio-voice-processing-deep-learning/).
- Warden, Pete. "Cross-compiling TensorFlow for the Raspberry Pi." Pete Warden's blog. August 20, 2017. Accessed December 03, 2017. [https://petewarden.com/2017/08/20/cross-compiling-tensorflow-for-the-raspberry-pi/](https://petewarden.com/2017/08/20/cross-compiling-tensorflow-for-the-raspberry-pi/).