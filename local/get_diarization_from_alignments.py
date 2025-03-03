import argparse
import os
import glob
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser("Parsing Forced alignments")
parser.add_argument("--in_file", type=str, default=r"F:\data\ami\falign\train\tri3a_train_ali")
parser.add_argument("--out_file", type=str, default=r"data\diarization_train\$train.json")

def parse_falign_file(falign_file):

    diarization = {}
    with open(falign_file, "r") as f:
        for line in f:
            start, stop, meta = line.split("\t")
            start = int(float(start)*16000)
            stop = int(float(stop)*16000)
            spk = re.findall("(spk[0-9]+)", meta)[0]
            if spk not in diarization.keys():
                diarization[spk] = [[start, stop]]
            else:
                diarization[spk].append([start, stop])


    return diarization

if __name__ == "__main__":
    args = parser.parse_args()

    os.makedirs(Path(args.out_file).parent, exist_ok=True)
    falign_sessions = glob.glob(os.path.join(args.in_file, "*.txt"))
    assert len(falign_sessions) > 0, "empty folder ?"

    diarization = {}
    for f in falign_sessions:
        sess = Path(f).stem.split("_")[-1]
        diarization[sess] = parse_falign_file(f)


    with open(args.out_file, "w") as f:
        json.dump(diarization, f, indent=4)
