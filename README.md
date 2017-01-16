# BADC-2017

#### Bird Audio Detection challenge submission using an ensemble of convolutional neural networks

---

Challenge info: http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/

> The task is to design a system that, given a short audio recording, returns a binary decision for the presence/absence of bird sound (bird sound of any kind).

---

Pretty straightforward implementation of convolutional detectors for this task, nothing too fancy. Data augmentation done with:

- noise corruption of training files,
- averaging of two training spectrograms at random,
- time-shifts of whole recordings,
- permutations of spectrogram segments.

Final submission consists of averaged predictions from 6 nets:

#### run-A

High resolution spectrogram convnet:
- trained on whole dataset,
- 5 conv/pooling LeakyReLU layers with batch normalization and delta channel,
- global max pooling before output.

![A](https://raw.githubusercontent.com/karoldvl/BADC-2017/master/results/A-filters.png)

#### run-B

Same as `run-A`, different random seed.

![B](https://raw.githubusercontent.com/karoldvl/BADC-2017/master/results/B-filters.png)

#### run-A_ds

Same as `run-A`, but low resolution (downsampled).

![A_ds](https://raw.githubusercontent.com/karoldvl/BADC-2017/master/results/A_ds-filters.png)

#### run-B_ds

Same as `run-A_ds`, different random seed.

![B_ds](https://raw.githubusercontent.com/karoldvl/BADC-2017/master/results/B_ds-filters.png)

#### run-23

Low resolution spectrogram convnet.

When compared to runs `A_ds`/`B_ds`:
- less filters,
- no L2 regularization,
- more epochs,
- faster learning rate decay,
- less dropout.

Originally used shuffle_spec() with 5 segments (spec length of 475) and less noise augmentations to generate the predictions.

![23](https://raw.githubusercontent.com/karoldvl/BADC-2017/master/results/23-filters.png)

#### run-26

High resolution spectrogram convnet.

When compared to runs `A`/`B`:
- more filters,
- lower regularization,
- lower learning rate,
- more epochs.

![26](https://raw.githubusercontent.com/karoldvl/BADC-2017/master/results/26-filters.png)
