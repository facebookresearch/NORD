#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import librosa
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from scipy.special import softmax
from tqdm import tqdm

from models import waveform_model_tracking


def __load_model(ckpt_path: str, nmr: int = 1, ch: int = 3, device=None) -> nn.Module:
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


def run_eval(model, ref_sig: torch.Tensor, test_sig: torch.Tensor):
    """Run NORD model infernce to evaluate the relative depth between two binaural (+ mono) speech recordings

    Args:
        model (model): a pre-trained model
        ref_sig (torch.Tensor): reference signal
        test_sig (torch.Tensor): test signal
    """
    with torch.no_grad():
        pref, quantf = model.forward(ref_sig, test_sig)
        # print(pref.detach().squeeze(0).cpu().numpy().shape)
        # print(quantf.detach().squeeze(0).cpu().numpy().shape)

    A = quantf.detach().squeeze(0).cpu().numpy()
    intervals_range = np.logspace(0, 1, base=6, num=20) - 1
    relative_depth = intervals_range[np.argmax(softmax(A.mean(1), axis=0))]
    return pref, relative_depth


def load_waveform_3ch(
    ref_bin: str, test_bin: str, ref_mono: str, test_mono: str, sr: int = 16000
):
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
    audio_ref, fs = librosa.load(ref_bin, mono=False, sr=sr)
    audio_test, fs = librosa.load(test_bin, mono=False, sr=sr)

    audio1_mono, fs = librosa.load(ref_mono, mono=False, sr=sr)
    audio2_mono, fs = librosa.load(test_mono, mono=False, sr=sr)

    audio1_mono = np.expand_dims(audio1_mono, axis=0)
    audio2_mono = np.expand_dims(audio2_mono, axis=0)

    ref, test = np.concatenate((audio_ref, audio1_mono)), np.concatenate(
        (audio_test, audio2_mono)
    )

    return ref, test


def load_waveform_2ch(ref_bin: str, test_bin: str, sr: int = 16000):
    """Load binaural audio files

    Args:
        ref_bin (str): reference binaural file path
        test_bin (str): test binaural file path

    Returns:
        np.ndarray: return 2 channel waveform
    """
    # load recordings
    audio_ref, fs = librosa.load(ref_bin, mono=False, sr=sr)
    audio_test, fs = librosa.load(test_bin, mono=False, sr=sr)
    return ref, test


def __test_with_rand(ch=3):
    ref = torch.randn(1, ch, 16000)
    test = torch.randn(1, ch, 16000)
    return ref, test


base_config = OmegaConf.create(
    {
        "gpu_id": 0,
        "ckpt_path": "saved_model_ckpt/scratchJNDdefault_best_model.pth",
        "nmr": 1,
        "num_ch": 3,
        "eval_config": "eval_config.yaml",
        "eval_list": None,
        "sampling_rate": 16000,
    }
)

###############################################################################
### sampling_rate = 16000Hz is fixed and waveform data should be preprocessed accordingly

if __name__ == "__main__":
    conf_cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(base_config, conf_cli)
    conf.eval_list = OmegaConf.load(conf.eval_config)

    device = torch.device(
        "cuda:{}".format(conf.gpu_id) if torch.cuda.is_available() else "cpu"
    )
    if conf.num_ch == 3:
        # load a pre-trained model
        model = __load_model(
            ckpt_path=conf.ckpt_path, nmr=conf.nmr, ch=3, device=device
        )
        nitems = len(conf.eval_list.evaluation_files)
        print(f"Found {nitems} items in eval_config")
        #
        pref_all = np.full((nitems, 2), np.nan)
        relative_depth_all = np.full((nitems, 1), np.nan)
        for i in tqdm(range(nitems), desc="Processing"):
            ref_bin, test_bin, ref_mono, test_mono = (
                conf.eval_list.evaluation_files[i].ref_bin,
                conf.eval_list.evaluation_files[i].test_bin,
                conf.eval_list.evaluation_files[i].ref_mono,
                conf.eval_list.evaluation_files[i].test_mono,
            )
            ref, test = load_waveform_3ch(ref_bin, test_bin, ref_mono, test_mono)
            pref, relative_depth = run_eval(model, ref, test)
            pref_all[i, :], relative_depth_all[i, :] = pref, relative_depth
            # print(f"Preference Score:{pref}, Relative Depth: {relative_depth}")
    elif conf.num_ch == 2:
        # load a pre-trained model
        model = __load_model(
            ckpt_path=conf.ckpt_path, nmr=conf.nmr, ch=2, device=device
        )
        nitems = len(conf.eval_list.evaluation_files)
        print(f"Found {nitems} items in eval_config")
        #
        pref_all = np.full((nitems, 2), np.nan)
        relative_depth_all = np.full((nitems, 1), np.nan)
        for i in tqdm(range(nitems), desc="Processing"):
            ref_bin, test_bin = (
                conf.eval_list.evaluation_files[i].ref_bin,
                conf.eval_list.evaluation_files[i].test_bin,
            )
            ref, test = load_waveform_2ch(ref_bin, test_bin)
            pref, relative_depth = run_eval(model, ref, test)
            pref_all[i, :], relative_depth_all[i, :] = pref, relative_depth
            # print(f"Preference Score:{pref}, Relative Depth: {relative_depth}")
    else:
        raise f"Unsupported num_ch={conf.num_ch}. Currently Nord supports num_ch=2 (binaural only) or num_ch=3 (binauaral & mono)"
    # Debug
    # ref, test = __test_with_rand(ch=conf.num_ch)
    # pref, relative_depth = run_eval(model, ref, test)
    # print(f"Preference Score:{pref}, Relative Depth: {relative_depth}")
