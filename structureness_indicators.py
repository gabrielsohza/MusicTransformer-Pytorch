from glob import glob
from argparse import ArgumentParser
import numpy as np
from statistics import mean
from os.path import exists
from midi2audio import FluidSynth
from mueller_audio_tools.scapeplot import compute_fitness_scape_plot, normalization_properties_SSM
from mueller_audio_tools.ssm_features import compute_SM_from_filename, compute_tempo_rel_set
from tqdm.contrib.concurrent import process_map


# parameters for SSM computation
tempo_rel_set = compute_tempo_rel_set(0.5, 2, 7) # for tempo invariance
shift_set = np.array([x for x in range(12)])     # for tranposition invariance
rel_threshold = 0.25                             # the proportion of (highest) values to retain
penalty = -2                                     # all values below ``rel_threshold`` are set to this
shorts, mids, longs = [], [], []

def compute_piece_ssm_scplot(wav_path):
    # compute & save self-similarity matrix (default resulting sample rate: 2Hz)
    _, _, _, _, S, _ = compute_SM_from_filename(
      wav_path, 
      tempo_rel_set=tempo_rel_set, 
      shift_set=shift_set, 
      thresh=rel_threshold,
      penalty=penalty
    )
    S = normalization_properties_SSM(S)

    # compute & save fitness scape plot
    SP = compute_fitness_scape_plot(S)[0]
    np.save(wav_path[:-3] + "npy", SP)


def compute_structure_indicator(fitness, low_bound_sec=0, upp_bound_sec=128, sample_rate=2):
    '''
    Computes the structureness indicator SI(low_bound_sec, upp_bound_sec) from fitness scape plot (stored in a MATLAB .mat file).
    (Metric ``SI``)
    Parameters:
    mat_file (str): path to the .mat file containing fitness scape plot of a piece. (computed by ``run_matlab_scapeplot.py``).
    low_bound_sec (int, >0): the smallest timescale (in seconds) you are interested to examine.
    upp_bound_sec (int, >0): the largest timescale (in seconds) you are interested to examine.
    sample_rate (int): sample rate (in Hz) of the input fitness scape plot.
    Returns:
    float: 0~1, the structureness indicator (i.e., max fitness value) of the piece within the given range of timescales.
    '''
    assert low_bound_sec > 0 and upp_bound_sec > 0, '`low_bound_sec` and `upp_bound_sec` should be positive, got: low_bound_sec={}, upp_bound_sec={}.'.format(low_bound_sec, upp_bound_sec)
    low_bound_ts = int(low_bound_sec * sample_rate) - 1
    upp_bound_ts = int(upp_bound_sec * sample_rate)
    f_mat = np.load(fitness)
    score = 0 if low_bound_ts >= f_mat.shape[0] else np.max(f_mat[ low_bound_ts : upp_bound_ts ])

    return score


def SIs_from_mid(mid):
    extension_split = mid.rfind(".")
    filepath, extension = mid[:extension_split], mid[extension_split:]

    wav_path = f"{filepath}.wav"
    if not exists(wav_path):
        FluidSynth().midi_to_audio(mid, wav_path)

    fit_path = f"{filepath}.npy"
    if not exists(fit_path):
        compute_piece_ssm_scplot(wav_path)
    
    short = compute_structure_indicator(fit_path, 3, 8)
    mid   = compute_structure_indicator(fit_path, 8, 15)
    long  = compute_structure_indicator(fit_path, 15)
    return (short, mid, long)

def structureness_indicators(plots_dir, max_workers=10, chunksize=1):
    midis = glob(f"{plots_dir}/**/rand-*.[mid|MID]*", recursive=True)
    
    results = process_map(SIs_from_mid, midis, max_workers=max_workers, chunksize=chunksize)

    print(f"Mean short SI: {mean([r[0] for r in results])}")
    print(f"Mean mid   SI: {mean([r[1] for r in results])}")
    print(f"Mean long  SI: {mean([r[2] for r in results])}")