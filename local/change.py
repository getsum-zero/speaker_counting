import os
import re
import soundfile as sf
import librosa


file = "/home/getsum/data/DIHARD3/preprocess/eval/wav/DH_EVAL_0160.wav"
data, samplerate = sf.read(file)
print(data.shape, samplerate)

file = "/home/getsum/data/dihard3_dec/Denoising/1_DH_EVAL_0160_(Vocals).mp3"
data, samplerate = sf.read(file)
print(data.shape, samplerate)


directory = "/home/getsum/data/dihard3_dec/Denoising"
save_directory = "/home/getsum/data/dihard3_dec/new"

for filename in os.listdir(directory):
    if "Vocals" in filename:
        keylist = filename.replace("(Vocals)", "").split(".")[0].split("_")
        new_filename = keylist[1] + "_" + keylist[2] + "_" + keylist[3] + ".wav"


        data, samplerate = sf.read(os.path.join(directory, filename))
        data_mono = data.mean(axis=1) if data.ndim > 1 else data

        data_resampled = librosa.resample(data_mono, orig_sr=samplerate, target_sr=16000)
        sf.write(os.path.join(save_directory, new_filename), data_resampled, 16000)

        data, samplerate = sf.read(os.path.join(save_directory, new_filename))
        print(data.shape, samplerate)
    