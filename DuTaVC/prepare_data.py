import os
import random
import tgt
from scipy.stats import mode
import librosa
from librosa.core import load
from librosa.filters import mel as librosa_mel_fn
mel_basis = librosa_mel_fn(22050, 1024, 80, 0, 8000)
import pickle
import numpy as np
import torch
use_gpu = torch.cuda.is_available()
import sys
sys.path.append('speaker_encoder/')
from encoder import inference as spk_encoder
from pathlib import Path
import multiprocessing
import shutil
from tqdm import tqdm
import h5py
import csv
import re
from num2words import num2words

def get_mel(index, savepath, commands):
    hf_m = h5py.File(os.path.join(savepath, 'mels', str(index)+ '.npy'), 'w')
    hf_w = h5py.File(os.path.join(savepath, 'wavs', str(index)+ '.npy'), 'w')
    for wavpath, textpath in tqdm(commands):
        wav, _ = load(wavpath, sr=22050)
        wav = wav[:(wav.shape[0] // 256)*256]
        wav = np.pad(wav, 384, mode='reflect')
        hf_w.create_dataset(wavpath.split('/')[-1].replace('.wav', ''), data=wav)
        stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
        stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
        mel_spectrogram = np.matmul(mel_basis, stftm)
        log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
        hf_m.create_dataset(wavpath.split('/')[-1].replace('.wav', ''), data=log_mel_spectrogram)
        # print(wavpath)
    hf_m.close()
    hf_w.close()

def get_textgrid(textpath, savepath):
    shutil.copyfile(textpath, os.path.join(savepath, 'textgrids', textpath.split('/')[-1]))
    print(textpath)

def get_embed(f, datapath, spk_encoder, savepath):
    if not os.path.exists(os.path.join(savepath, f)):
        hf = h5py.File(os.path.join(datapath, f), 'r')
        hf_e = h5py.File(os.path.join(savepath, f), 'w')
        allkeys = hf.keys()
        for k in tqdm(allkeys):
            wav = np.array(hf.get(k))
            wav_preprocessed = spk_encoder.preprocess_wav(wav)
            embed = spk_encoder.embed_utterance(wav_preprocessed)
            hf_e.create_dataset(k, data=embed)
        hf.close()
        hf_e.close()

# exclude utterances where MFA couldn't recognize some words
def exclude_spn(textgrid):
    t = tgt.io.read_textgrid(textgrid)
    t = t.get_tier_by_name('phones')
    spn_found = False
    for i in range(len(t)):
        if t[i].text == 'spn':
            spn_found = True
            break
    if not spn_found:
        return True
    return False

def makedata_aty():
    txtpath = './DuTaVC/atydata/cleandata/textgrids'
    audiopath = './DuTaVC/atydata/cleandata/Audios_22050'
    savepath = './DuTaVC/aty_data'
    allspks = ['0005']
    for spk in allspks:
        os.makedirs(os.path.join(savepath, 'wavs', spk), exist_ok=True)
        os.makedirs(os.path.join(savepath, 'textgrids', spk), exist_ok=True)
        for root, dir, files in os.walk(os.path.join(txtpath, spk)):
            for f in tqdm(files):
                if f.endswith('.TextGrid'):
                    if exclude_spn(os.path.join(root, f)):
                        shutil.copyfile(os.path.join(txtpath, spk, f), os.path.join(savepath, 'textgrids', spk, f))
                        shutil.copyfile(os.path.join(audiopath, spk, f.replace('.TextGrid', '.wav')), os.path.join(savepath, 'wavs', spk, f.replace('.TextGrid', '.wav')))

def get_mel_atypical(wav_path, save_path):
    try:
        wav, _ = load(wav_path, sr=22050)
        wav = wav[:(wav.shape[0] // 256)*256]
        wav = np.pad(wav, 384, mode='reflect')
        stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
        stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
        mel_spectrogram = np.matmul(mel_basis, stftm)
        log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
        with open(save_path, 'wb') as f:
            np.save(f, log_mel_spectrogram)
    except:
        print(wav_path)


def load_data_aty():
    audiopath = './DuTaVC/aty_data/wavs'
    savepath = './DuTaVC/aty_data/'
    allspks = ['0005']

    commands = []
    for spk in allspks:
        os.makedirs(os.path.join(savepath, 'mels', spk), exist_ok=True)
        for root, dir, files in os.walk(os.path.join(audiopath, spk)):
            for f in files:
                if f.endswith('.wav'):
                    commands.append((os.path.join(root, f), os.path.join(savepath, 'mels', spk, f.replace('.wav', '.npy'))))
    print(len(commands))
    with multiprocessing.Pool(processes=40) as pool:
        pool.starmap(get_mel_atypical, commands)


def generate_avg_aty():
    savepath = './DuTaVC/aty_data/'
    mfapath = './DuTaVC/aty_data/textgrids'
    pkl_path = './DuTaVC/libritts_data/mels_mode.pkl'
    allspks = ['0005']
    with open(pkl_path, 'rb') as f:
        mels_mode = pickle.load(f)

    for spk in allspks:
        os.makedirs(os.path.join(savepath, 'mels_mode', spk), exist_ok=True)
        for root, dir, files in os.walk(os.path.join(savepath, 'mels', spk)):
            for f in files:
                if f.endswith('.npy'):
                    textgrid = os.path.join(mfapath, spk, f.replace('.npy', '.TextGrid'))
                    t = tgt.io.read_textgrid(textgrid)
                    m = np.load(os.path.join(root, f))
                    m_mode = np.full((m.shape[0], m.shape[1]), np.log(1e-5))
                    t = t.get_tier_by_name('phones')
                    for i in range(len(t)):
                        phoneme = t[i].text
                        start_frame = int(t[i].start_time * 22050.0) // 256
                        end_frame = int(t[i].end_time * 22050.0) // 256 + 1
                        if end_frame > m_mode.shape[1]:
                            end_frame = m_mode.shape[1]
                        print(f, phoneme, m_mode.shape, mels_mode[phoneme].shape, start_frame, end_frame)
                        m_mode[:, start_frame:end_frame] = np.repeat(np.expand_dims(mels_mode[phoneme], 1),
                                                                    end_frame - start_frame, 1)
                    np.save(os.path.join(savepath, 'mels_mode', spk, f.replace('.npy', '_avgmel.npy')), m_mode)


def get_embed(wav_path, spk_encoder, savepath):
    if not os.path.exists(savepath):
        wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
        embed = spk_encoder.embed_utterance(wav_preprocessed)
        np.save(savepath, embed)
        # print(savepath)

def generate_emb_aty():
    datapath = './DuTaVC/aty_data/wavs/'
    savepath = './DuTaVC/aty_data/'
    allspks = ['0005']
    # loading speaker encoder
    enc_model_fpath = Path('checkpts/spk_encoder/pretrained.pt')  # speaker encoder path
    spk_encoder.load_model(enc_model_fpath, device="cpu")
    cmds = []
    for spk in allspks:
        os.makedirs(os.path.join(savepath, 'embeds', spk), exist_ok=True)
        for root, dir, files in os.walk(os.path.join(datapath, spk)):
            for f in files:
                if f.endswith('.wav'):
                    savename = os.path.join(savepath, 'embeds', spk, f.replace('.wav', '.npy'))
                    cmds.append((os.path.join(root, f), spk_encoder, savename))
    print(len(cmds))
    random.shuffle(cmds)
    for c in tqdm(cmds):
        get_embed(c[0], c[1], c[2])


def cal_mean_std_aty():
    datapath = './DuTaVC/aty_data/'
    allspks = ['0005']
    for spk in allspks:
        os.makedirs(os.path.join(datapath, 'stats', spk), exist_ok=True)
        data_list = []
        for root, dir, files in os.walk(os.path.join(datapath, 'mels', spk)):
            for f in files:
                if f.endswith('.npy'):
                    data_list.append(os.path.join(root, f))
        scp_list_source = data_list
        print(len(scp_list_source))
        feaNormCal = []
        for s in tqdm(scp_list_source):
            x = np.load(s).transpose(1, 0)
            feaNormCal.extend(x)
        nFrame = np.shape(feaNormCal)[0]
        print(nFrame)
        feaMean = np.mean(feaNormCal, axis=0)
        for i in range(nFrame):
            if i == 0:
                feaStd = np.square(feaNormCal[i] - feaMean)
            else:
                feaStd += np.square(feaNormCal[i] - feaMean)
        feaStd = np.sqrt(feaStd / nFrame)
        result = np.vstack((feaMean, feaStd))
        np.savetxt(os.path.join(datapath, 'stats', spk, 'global_mean_var.txt'), result)

def cal_avg_phonemetime_aty():
    spks = ['0005']
    mfapath = './DuTaVC/aty_data/textgrids/'
    savepath = './DuTaVC/aty_data/stats/'
    lpath = './DuTaVC/libritts_data/phonemes.pkl'
    with open(lpath, 'rb') as f:
        ls_dict = pickle.load(f)
    phoneme_list = ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2',
                    'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0',
                    'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH',
                    'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2',
                    'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2',
                    'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG',
                    'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P',
                    'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2',
                    'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH', 'sil', 'sp', 'spn']
    for spk in spks:
        phoneme_dict = {}
        for p in phoneme_list:
            phoneme_dict[p] = {'duration':0., 'number':0}
        for root, dir, files in os.walk(os.path.join(mfapath, spk)):
            for f in files:
                if f.endswith('.TextGrid'):
                    t = tgt.io.read_textgrid(os.path.join(root, f))
                    t = t.get_tier_by_name('phones')
                    for i in range(len(t)):
                        phoneme = t[i].text
                        phoneme_dict[phoneme]['duration'] += t[i].end_time - t[i].start_time
                        phoneme_dict[phoneme]['number'] += 1
        for p in phoneme_list:
            if phoneme_dict[p]['number'] > 0:
                phoneme_dict[p]['avg_duration'] = float(phoneme_dict[p]['duration'] / phoneme_dict[p]['number'])
            else:
                phoneme_dict[p]['avg_duration'] = ls_dict[p]['avg_duration']
        print(phoneme_dict)
        os.makedirs(os.path.join(savepath, spk), exist_ok=True)
        with open(os.path.join(savepath, spk, 'phonemes.pkl'), 'wb') as f:
            pickle.dump(phoneme_dict, f)


if __name__ == "__main__":
    print('test')
    makedata_aty()
    load_data_aty()
    generate_avg_aty()
    generate_emb_aty()
    cal_mean_std_aty()
    cal_avg_phonemetime_aty()
