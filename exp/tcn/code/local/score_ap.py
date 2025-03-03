import argparse
import os
import json
import glob
from pathlib import Path
import re
import json
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, confusion_matrix
import pandas as pd
import numpy as np
from scipy.signal import medfilt

parser = argparse.ArgumentParser("scoring")
parser.add_argument("--preds_dir", type=str, default=r"./exp/tcn/preds/checkpoints/epoch=97-step=59388.ckpt")
parser.add_argument("--diarization_file", type=str, default=r"./data/diarization_eval/eval.json")


def build_target_vector(sess_diarization, subsample=160):

    # get maxlen
    maxlen = max([sess_diarization[spk][-1][-1] for spk in sess_diarization.keys()])
    dummy = np.zeros(maxlen//subsample, dtype="uint8")

    for spk in diarization[sess].keys():
        if spk == "garbage":
            continue
        for s, e in diarization[sess][spk]:
            s = int(s/subsample)
            e = int(np.ceil(e/subsample))
            dummy[s:e] += 1
    return dummy


def one_hot(a, num_classes):

    a = np.clip(a, 0, num_classes - 1)
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def score(preds, target_vector):
    target_vector[target_vector >= 2] = 2
    target_vector_cat = target_vector
    target_vector = one_hot(target_vector, preds.shape[0])
    minlen = min(target_vector.shape[0], preds.shape[-1])
    target_vector = target_vector[:minlen, :]
    target_vector_cat = target_vector_cat[:minlen]
    preds = preds[:, :minlen].T

    count_ap = average_precision_score(target_vector, preds, average=None)
    osd_ap = average_precision_score(target_vector_cat >= 2, np.sum(preds[:, 2:], -1))
    vad_ap = average_precision_score(target_vector_cat >= 1, np.sum(preds[:, 1:], -1))

    preds_cat = np.argmax(preds, axis=1)
    cm = confusion_matrix(target_vector_cat, preds_cat)

    return count_ap, vad_ap, osd_ap, cm


# def score(preds, target_vector):
#     target_vector = np.where(target_vector >= 2, 1, 0)
#     target_vector_cat = target_vector
#     target_vector = one_hot(target_vector, preds.shape[0])
#     minlen = min(target_vector.shape[0], preds.shape[-1])
#     target_vector = target_vector[:minlen, :]
#     target_vector_cat = target_vector_cat[:minlen]
#     preds = preds[:, :minlen].T
#     max_indices_per_row = np.argmax(preds, axis=1)
#     one_hot_encoded = np.zeros_like(preds)
#     one_hot_encoded[np.arange(preds.shape[0]), max_indices_per_row] = 1
#
#     y_true = np.argmax(target_vector, axis=1)
#     y_pred = np.argmax(one_hot_encoded, axis=1)
#     y_pred = np.where(y_pred >= 2, 1, 0)
#
#     count_ap = average_precision_score(target_vector, preds, average=None)
#     count_p = precision_score(y_true, y_pred, average=None)
#     count_r = recall_score(y_true, y_pred, average=None)
#     osd_ap = average_precision_score(target_vector_cat >= 2, np.sum(preds[:, 2:], -1))
#     osd_p = precision_score(y_true, y_pred)
#     osd_r = recall_score(y_true, y_pred)
#     vad_ap = average_precision_score(target_vector_cat >= 1, np.sum(preds[:, 1:], -1))
#     vad_p = precision_score(y_true, y_pred)
#     vad_r = recall_score(y_true, y_pred)
#
#     return count_ap, count_p, count_r, vad_ap, vad_p, vad_r, osd_ap, osd_p, osd_r


def process_preds(preds):

    # average all from same session
    # apply medfilter
    mat = []
    minlen = np.inf
    for i in preds:
        tmp = np.load(i)[0]
        minlen = min(minlen, tmp.shape[-1])
        mat.append(tmp)

    mat = [x[:minlen] for x in mat]
    mat = np.mean(np.stack(mat), 0)

    return mat #medfilt(mat, 5)

if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.diarization_file, "r") as f:
        diarization = json.load(f)

    preds_hash = {}
    preds = glob.glob(os.path.join(args.preds_dir, "*.npy"))

    for p in preds:
        session = Path(p).stem.split(".")[0]
        if session not in preds_hash.keys():
            preds_hash[session] = [p]
        else:
            preds_hash[session].append(p)       

    scores = {}
    total = np.zeros((3, 3))
    for sess in diarization.keys():
        if sess not in preds_hash.keys():
            continue

        target_vector = build_target_vector(diarization[sess])
        preds = preds_hash[sess]
        preds = process_preds(preds)
        count_ap, vad_ap, osd_ap, cm = score(preds, target_vector)
        if cm.shape != (3, 3):
            padded_arr = np.zeros((3, 3))
            padded_arr[:cm.shape[0], :cm.shape[1]] = cm
        else:
            padded_arr = cm
        total += padded_arr
        scores[sess] = {"count_ap": count_ap, "vad_ap": vad_ap, "osd_ap": osd_ap,
                        "one_speaker": [padded_arr[1, 1], np.sum(padded_arr, axis=1)[1]],  
                        "more_than_one_speakers": [padded_arr[2, 2], np.sum(padded_arr, axis=1)[2]]
                         }
        # count_ap, count_p, count_r, vad_ap, vad_p, vad_r, osd_ap, osd_p, osd_r = score(preds, target_vector)
        # scores[sess] = {"count_ap": count_ap, "vad_ap": vad_ap, "osd_ap": osd_ap, "count_p": count_p, "count_r": count_r, "vad_p": vad_p, "vad_r": vad_r,  "osd_p": osd_p, "osd_r": osd_r}

    dt = pd.DataFrame.from_dict(scores)
    # scores["TOTAL"] = {"count_ap": dt.iloc[0, :].mean(), "vad_ap": dt.iloc[1, :].mean(), "osd_ap": dt.iloc[2, :].mean()}
    # scores["TOTAL"] = {"count_ap": dt.iloc[0, :].mean(), "vad_ap": dt.iloc[1, :].mean(), "osd_ap": dt.iloc[2, :].mean(), "count_p": dt.iloc[3, :].mean(), "count_r": dt.iloc[4, :].mean(), "vad_p": dt.iloc[5, :].mean(), "vad_r": dt.iloc[6, :].mean(), "osd_p": dt.iloc[7, :].mean(), "osd_r": dt.iloc[8, :].mean()}
    dt = pd.DataFrame.from_dict(scores).to_json(os.path.join(args.preds_dir, "APs.json"))
    print(total / np.sum(total, axis=1).reshape(-1, 1))
    print(total)
    sum = 0
    for i in range(3):
        sum = sum + total[i][i]
    print(sum / np.sum(total))














