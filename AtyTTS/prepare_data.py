import os
from tqdm import tqdm
from librosa.core import load
import soundfile as sf
import os
import multiprocessing
import csv
import random
import pandas as pd
import sys
sys.path.append('./hifi-gan/')
from meldataset import mel_spectrogram
import torchaudio as ta
import numpy as np

n_fft=1024 
n_mels=80
sample_rate=22050
hop_length=256
win_length=1024
f_min=0.
f_max=8000
debug = False

def generate_data_aty(spk):
    lines = []
    datapath = './atypicalspeech/Audios_22050/'+spk
    savepath_mel = './atypicalspeech/mel/'+spk
    os.makedirs(savepath_mel, exist_ok=True)
    for root, dir, files in os.walk(datapath):
        for f in files:
            if f.endswith('.wav'):
                lines.append(os.path.join(root, f))
    if debug:
        lines = lines[-3:]
    for line in tqdm(lines):
        filepath = line
        audio, sr = ta.load(filepath)
        mel = mel_spectrogram(audio, n_fft, n_mels, sr, hop_length, win_length, f_min, f_max, center=False)
        mel = mel.squeeze().numpy()
        mel_filename = os.path.join(savepath_mel, filepath.split('/')[-1].replace('.wav', '.npy'))
        np.save(mel_filename, mel)

def resample_one(source, target):
    wav, _ = load(source, sr=22050)
    sf.write(target, wav, 22050, 'PCM_32')
    print(target)

def resample():
    audiopath = './atypicalspeech/Audios'
    savepath = './atypicalspeech/Audios_22050'
    spks = ['0005']
    cmds = []
    for spk in spks:
        os.makedirs(os.path.join(savepath, spk), exist_ok=True)
        for root, dir, files in os.walk(os.path.join(audiopath, spk)):
            for f in files:
                if f.endswith('.wav'):
                    cmds.append((os.path.join(root, f), os.path.join(savepath, spk, f)))
    print(len(cmds))
    with multiprocessing.Pool(processes=50) as pool:
        pool.starmap(resample_one, cmds)

def makelist():
    wavpath = './atypicalspeech/Audios_22050'
    spks = ['0005']
    metapath = './atypicalspeech/excel_out_slurp_metadata_named.csv'
    savepath = './AtyTTS/resources/filelists/atypical'
    lines = []
    with open(metapath, mode ='r') as file:
        csvFile = csv.reader(file) 
        for line in csvFile: 
            lines.append(line)
    lines = lines[1:]
    metadict = {}
    for line in lines:
        metadict[line[2]] = line[3]
    metapath_fsc = './atypicalspeech/excel_out.xlsx'
    meta = pd.read_excel(metapath_fsc)
    filenames = meta['Filename']
    Texts = meta['Text']
    for i in range(len(filenames)):
        metadict[filenames[i]] = Texts[i]
    for spk in spks:
        os.makedirs(os.path.join(savepath, spk), exist_ok=True)
        alltargets = []
        for root, dir, files in os.walk(os.path.join(wavpath, spk)):
            for f in tqdm(files):
                if f.endswith('.wav'):
                    name = 'XXXX_' + f.split('_', 1)[-1]
                    if name in metadict.keys():
                        alltargets.append(os.path.join(root, f) + '|' + metadict[name]+'\n')
                    name = 'XXX_' + f.split('_', 1)[-1]
                    if name in metadict.keys():
                        alltargets.append(os.path.join(root, f) + '|' + metadict[name]+'\n')
        with open(os.path.join(savepath, spk, 'all.txt'), 'w') as fi:
            fi.writelines(alltargets)

def splitlist():
    spks = ['0005']
    savepath = './AtyTTS/resources/filelists/atypical'
    for spk in spks:
        with open(os.path.join(savepath, spk, 'all.txt'), 'r') as fi:
            lines = fi.readlines()
        random.shuffle(lines)
        num = len(lines) // 10
        testlines = lines[:num]
        trainlines = lines[num:]
        with open(os.path.join(savepath, spk, 'train.txt'), 'w') as fi:
            fi.writelines(trainlines)
        with open(os.path.join(savepath, spk, 'test.txt'), 'w') as fi:
            fi.writelines(testlines)

def prepare_dataaug_list():
    spks = ['0005']
    savepath = './AtyTTS/resources/filelists/atypical'
    datapath = './DuTaVC/results_dataug_fortts'
    for spk in spks:
        outs = []
        with open(os.path.join(savepath, spk, 'all.txt'), 'r') as fi:
            lines = fi.readlines()
        print(len(lines))
        targets = []
        for root, dir, files in os.walk(os.path.join(datapath, 'aty', spk)):
            for f in files:
                if f.endswith('.wav'):
                    if os.path.exists(os.path.join(datapath, 'aty', spk, f)) and os.path.exists(os.path.join(datapath, 'source', spk, f)):
                        targets.append(f)
        print(len(targets))
        for f in targets:
            source = random.choice(lines).split('\n')[0]
            outs.append(source+'|'+os.path.join(datapath, 'aty', spk, f)+'|'+os.path.join(datapath, 'ty', spk, f)+'|'+os.path.join(datapath, 'source', spk, f)+'\n')
        with open(os.path.join(savepath, spk, 'all_aug.txt'), 'w') as fi:
            fi.writelines(outs)
        random.shuffle(outs)
        with open(os.path.join(savepath, spk, 'train_aug.txt'), 'w') as fi:
            fi.writelines(outs[4:])
        with open(os.path.join(savepath, spk, 'test_aug.txt'), 'w') as fi:
            fi.writelines(outs[:4])

# resample()
# makelist()
# splitlist()

# spks = ['0005', '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014',
#         '0015', '0017', '0018', '0019', '0020', '0021', '0022', '0023',
#         '0024', '0025', '0026']
# for spk in spks:
#     generate_data_aty(spk)
prepare_dataaug_list()