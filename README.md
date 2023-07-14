<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/80x15.png" /></a> This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

# Nord: Non-Matching Reference Based Relative Depth Estimation from Binaural Speech
[Link to the ICASSP 2023 Paper](https://ieeexplore.ieee.org/document/10094615).

This is an open-source release for the paper published in ICASSP 2023 conference: <b>"NORD: Non-Matching Reference Based Relative Depth Estimation from Binaural Speech" </b>. NORD is a novel framework for estimating the relative depth between two binaural speech recordings. In contrast to existing depth estimation techniques, ours only requires audio signals as input. We trained the framework to solve depth preference (i.e. which input perceptually sounds closer to the listener’s head), and quantification tasks (i.e. quantifying the depth difference between the inputs). In addition, training leverages recent advances in metric and multi-task learning, which allows the framework to be invariant to both signal content (i.e. non-matched reference) and directional cues (i.e. azimuth and elevation). Our framework has additional useful qualities that make it suitable for use as an objective metric to benchmark binaural audio systems, particularly depth perception and sound externalization.

This repo provides examples how to use the pre-trained model to evalute the relative depth between two binaural speech recordings.

## Requirements
Our code has been primarily tested on Ubuntu 20, but it should work on other versions OS that can support Python 3.8+ and PyTorch=1.10+. We strongly recommend using Anaconda or Miniconda for setting up the Python environment.
```
$ conda create --name nord_env --file requirements.txt --channel conda-forge --channel defaults
$ conda activate nord_env
## You will need to install PyTorch=1.10+, https://pytorch.org/
```

## Running Nord
### Comparing two binaural recordings
```
$ python evaluate.py num_ch=2 eval_config="eval_config.yaml"
```
### Comparing two binaural recordings which also have the mono signals.
This is typically the case when binaural signals are created/generated using HRTF or other binaural synsthesis techniques.
```
$ python evaluate.py num_ch=3 eval_config="eval_config.yaml"
```
Please refer to the "eval_config.yaml" to properly organize audio files for comparison.


## License
This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>, as found in the LICENSE file.

## Citation
If you find this repository useful in your research, please consider giving a star ⭐ and cite our ICASSP 2023 paper by using the following BibTeX entrys.
```
@INPROCEEDINGS{10094615,
  author={Manocha, Pranay and Gebru, Israel D. and Kumar, Anurag and Markovic, Dejan and Richard, Alexander},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Nord: Non-Matching Reference Based Relative Depth Estimation from Binaural Speech},
  year={2023},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10094615}}
```
