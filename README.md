# BADC 2017

#### Bird Audio Detection challenge submission using an ensemble of convolutional neural networks

---

Challenge info: http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/

---

Final submission consists of averaged predictions from 6 nets:

#### run-A

High resolution spectrogram convnet:
- trained on whole dataset
- 5 conv/pooling LeakyReLU layers with batch normalization and delta channel
- global max pooling before output

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
- Less filters
- No L2 regularization
- More epochs
- Faster learning rate decay
- Less dropout

Originally used shuffle_spec() with 5 segments (spec length of 475) and less noise augmentations.

![23](https://raw.githubusercontent.com/karoldvl/BADC-2017/master/results/23-filters.png)

#### run-26

High resolution spectrogram convnet.

When compared to runs `A`/`B`:
- More filters
- Lower regularization
- Lower learning rate
- More epochs

![26](https://raw.githubusercontent.com/karoldvl/BADC-2017/master/results/26-filters.png)
