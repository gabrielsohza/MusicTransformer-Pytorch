import torch
import torch.nn as nn
import os
import random
import pretty_midi
import pickle
import sys
from shutil import copyfile

from third_party.midi_processor.processor import encode_midi_original, encode_midi_modified, decode_midi_modified, decode_midi_original

from statistics import mean
from utilities.argument_funcs import parse_generate_args, print_generate_args
from model.music_transformer import MusicTransformer
from dataset.e_piano import create_epiano_datasets, compute_epiano_accuracy, process_midi
from torch.utils.data import DataLoader
from torch.optim import Adam

from utilities.constants import *
from utilities.device import get_device, use_cuda

from structureness_indicators import structureness_indicators

INDICES = []


# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Generates music from a model specified by command line arguments
    ----------
    """
    seed = random.randrange(sys.maxsize)
    print(f"Seed is {seed}")
    random.seed(seed)

    args = parse_generate_args()
    print_generate_args(args)

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")
    
    SEED = args.seed if args.seed is not None else random.randrange(sys.maxsize)
    print(f"Setting seed to {SEED}")
    random.seed(SEED)

    os.makedirs(args.output_dir, exist_ok=True)

    # Grabbing dataset if needed
    _, _, dataset = create_epiano_datasets(args.midi_root, args.num_prime, args.new_notation, random_seq=False)
    
    if(args.primer_file is not None):
        f = [args.primer_file]
    else:
        f = random.sample(range(len(dataset)), args.num_primer_files)

    for j in range(args.num_primer_files):
        idx = int(f[j])
        primer, _ = dataset[idx]
        primer = primer.int().to(get_device())

        print("Using primer index:", idx, "(", dataset.data_files[idx], ")")
        copyfile(dataset.data_files[idx], f"{args.output_dir}/original-{idx}.mid")

        model = MusicTransformer(new_notation=args.new_notation, n_layers=args.n_layers, num_heads=args.num_heads,
                    d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                    max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

        model.load_state_dict(torch.load(args.model_weights))

        f_path = os.path.join(args.output_dir, f"primer-{idx}.mid")
        if args.new_notation:
            decode_midi_modified(primer[:args.num_prime].tolist(), f_path)
        else:
            decode_midi_original(primer[:args.num_prime].tolist(), f_path)

        for i in range(args.num_samples):
            print(f"Generating song {i}")
            # GENERATION
            model.eval()
            with torch.set_grad_enabled(False):
                if(args.beam > 0):
                    print("BEAM:", args.beam)
                    beam_seq = model.generate(primer[:args.num_prime], args.target_seq_length, beam=args.beam)

                    f_path = os.path.join(args.output_dir, f"beam-{idx}-{i}.mid")
                    if args.new_notation:
                        decode_midi_modified(beam_seq[0].tolist(), f_path)
                    else:
                        decode_midi_original(beam_seq[0].tolist(), f_path)
                else:
                    print("RAND DIST")
                    rand_seq = model.generate(primer[:args.num_prime], args.target_seq_length, beam=0)

                    f_path = os.path.join(args.output_dir, f"rand-{idx}-{i}.mid")
                    if args.new_notation:
                        decode_midi_modified(rand_seq[0].tolist(), f_path)
                    else:
                        decode_midi_original(rand_seq[0].tolist(), f_path)

            print()

        #if args.struct:
        #    structureness_indicators(args.output_dir)
    
    print(f"Files were generated with seed {seed}")

if __name__ == "__main__":
    main()