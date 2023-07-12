import argparse
import os
import timeit

import librosa
import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax

from models import waveform_model_tracking


def __load_model(ckpt_path: str, nmr:int = 1, ch: int=3, device=None) -> nn.Module:
    """Loads a pytorch model from a given checkpoint

    Args:
        ckpt_path (str): full path to the checkpoint
        ch (int): number of channels
        device (torch.device): torch device

    Returns:
        nn.Module: : model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    model = waveform_model_tracking(nmr=nmr, ch=ch).to(device)
    print("Loading Checkpoint...", end="\x1b[1K\r")
    state = torch.load(
        ckpt_path, map_location="cpu"
    )  # change this and the above line to pretrain (state_base)
    model.load_state_dict(state["state_dict"], strict=False)
    print("Done Loading Checkpoint!")
    # Evaluation mode
    model.eval()
    return model


def run_eval(model, ref_sig, test_sig):
    with torch.no_grad():
        pref, quantf = model.forward(ref_sig, test_sig)

        print(pref.detach().squeeze(0).cpu().numpy().shape)
        print(quantf.detach().squeeze(0).cpu().numpy().shape)

    A = quantf.detach().squeeze(0).cpu().numpy()
    intervals_range = np.logspace(0, 1, base=6, num=20) - 1
    print(intervals_range[np.argmax(softmax(A.mean(1), axis=0))])


def load_waveform(ref_bin, test_bin, ref_mono, test_mono):
    """Load audio files and create three channel waveform arrays concatinating binaural and mono signals

    Args:
        ref_bin (str): reference binaural file path
        test_bin (str): test binaural file path
        ref_mono (str): reference mono file path
        test_mono (str): test mono file path

    Returns:
        np.ndarray: return 3 channel waveform
    """
    # load recordings
    audio_ref, fs = librosa.load(ref_bin, mono=False, sr=48000)
    audio_test, fs = librosa.load(test_bin, mono=False, sr=48000)

    audio1_mono, fs = librosa.load(ref_mono, mono=False, sr=48000)
    audio2_mono, fs = librosa.load(test_mono, mono=False, sr=48000)

    audio1_mono = np.expand_dims(audio1_mono, axis=0)
    audio2_mono = np.expand_dims(audio2_mono, axis=0)

    ref, test = np.concatenate((audio_ref, audio1_mono)), np.concatenate(
        (audio_test, audio2_mono)
    )

    return ref, test


def __test_with_rand():
    np.random.seed(12345)
    ref = torch.randn(1, 3, 16000)
    test = torch.randn(1, 3, 16000)
    return ref, test


###############################################################################
### PARSE SETTINGS ; sr = 16000Hz is fixed and waveform data should be preprocessed accordingly

parser = argparse.ArgumentParser()
parser.add_argument("--GPU_id", type=str, default="0")
parser.add_argument(
    "--ckpt_path", type=str, default="saved_model_ckpt/scratchJNDdefault_best_model.pth"
)
parser.add_argument("--nmr", type=int, default=1)
parser.add_argument("--num_ch", type=int, default=3)


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device(
        "cuda:{}".format(args.GPU_id) if torch.cuda.is_available() else "cpu"
    )
    # load a pre-trained model
    model = __load_model(ckpt_path=args.ckpt_path, nmr=args.nmr, ch=args.num_ch,device=device)
    # get reference and test waveforms
    ref, test = __test_with_rand()
    # run evaluation
    run_eval(model, ref, test)
