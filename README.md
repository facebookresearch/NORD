
# Nord: Non-Matching Reference Based Relative Depth Estimation from Binaural Speech

This is an open-source release for the paper published in ICASSP 2023 conference: " Nord: Non-Matching Reference Based Relative Depth Estimation from Binaural Speech". This code-base provides an example how to use the pre-trained model to evalute the relative depth between two binaural speech recordings.

### Abstract
NORD: is a novel framework for estimating the relative depth between two binaural speech recordings. In contrast to existing depth estimation techniques, ours only requires audio signals as input. We trained the framework to solve depth preference (i.e. which input perceptually sounds closer to the listener’s head), and quantification tasks (i.e. quantifying the depth difference between the inputs). In addition, training leverages recent advances in metric and multi-task learning, which allows the framework to be invariant to both signal content (i.e. non-matched reference) and directional cues (i.e. azimuth and elevation). Our framework has additional useful qualities that make it suitable for use as an objective metric to benchmark binaural audio systems, particularly depth perception and sound externalization. [[paper](https://ieeexplore.ieee.org/document/10094615]


## Requirements
This repository is tested on Ubuntu 20.04 using a V100 and the following settings.
- Python 3.8+
- PyTorch 1.10+

## Running Nord
```
# TODO Example
```

## License
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/80x15.png" /></a> This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

## Citation
If you find this repository useful in your research, please consider giving a star ⭐ and cite our ICASSP 2023 paper by using the following BibTeX entrys.
```
@INPROCEEDINGS{10094615,
  author={Manocha, Pranay and Gebru, Israel D. and Kumar, Anurag and Markovic, Dejan and Richard, Alexander},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Nord: Non-Matching Reference Based Relative Depth Estimation from Binaural Speech},
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10094615}}
```

## FAQ
- <b> Does the pre-trained model works on non-speech recordings? </b>
