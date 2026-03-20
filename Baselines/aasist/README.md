This directory contains the baseline code for **AASIST**.

A major portion of the code is adapted from the original implementation:
[https://github.com/clovaai/aasist](https://github.com/clovaai/aasist)

Minor modifications have been introduced to better suit our setup, including *on-the-fly augmentation* and *chunking* audio files into non-overlapping 4-second clips.

### Augmentations used

* `gaussian_noise_snr`
* `mp3_compression`
* `aliasing`
* `band_pass_filter`
* `gain`
* `gain_transitions`
* `loudness_norm`
* `speech_pertubation`
