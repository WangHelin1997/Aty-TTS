import argparse
import json
import os
import random

import numpy as np
import torchaudio
import torch
use_gpu = torch.cuda.is_available()
import librosa
from librosa.core import load
from librosa.filters import mel as librosa_mel_fn
mel_basis = librosa_mel_fn(22050, 1024, 80, 0, 8000)

import params
from model import DiffVC
import sys
sys.path.append('hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN
sys.path.append('speaker_encoder/')
from encoder import inference as spk_encoder
from tqdm import tqdm
import shutil

def get_mel(wav_path):
    wav, _ = load(wav_path, sr=22050)
    wav = wav[:(wav.shape[0] // 256)*256]
    wav = np.pad(wav, 384, mode='reflect')
    stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
    stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
    mel_spectrogram = np.matmul(mel_basis, stftm)
    log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
    return log_mel_spectrogram

def get_embed(wav_path):
    wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
    embed = spk_encoder.embed_utterance(wav_preprocessed)
    return embed

def noise_median_smoothing(x, w=5):
    y = np.copy(x)
    x = np.pad(x, w, "edge")
    for i in range(y.shape[0]):
        med = np.median(x[i:i+2*w+1])
        y[i] = min(x[i+w+1], med)
    return y

def mel_spectral_subtraction(mel_synth, mel_source, spectral_floor=0.02, silence_window=5, smoothing_window=5):
    mel_len = mel_source.shape[-1]
    energy_min = 100000.0
    i_min = 0
    for i in range(mel_len - silence_window):
        energy_cur = np.sum(np.exp(2.0 * mel_source[:, i:i+silence_window]))
        if energy_cur < energy_min:
            i_min = i
            energy_min = energy_cur
    estimated_noise_energy = np.min(np.exp(2.0 * mel_synth[:, i_min:i_min+silence_window]), axis=-1)
    if smoothing_window is not None:
        estimated_noise_energy = noise_median_smoothing(estimated_noise_energy, smoothing_window)
    mel_denoised = np.copy(mel_synth)
    for i in range(mel_len):
        signal_subtract_noise = np.exp(2.0 * mel_synth[:, i]) - estimated_noise_energy
        estimated_signal_energy = np.maximum(signal_subtract_noise, spectral_floor * estimated_noise_energy)
        mel_denoised[:, i] = np.log(np.sqrt(estimated_signal_energy))
    return mel_denoised


def inference(generator, src_path, tgt_path, save_path, mean, std, emb):

    # loading source and reference wavs, calculating mel-spectrograms and speaker embeddings
    mel_target = torch.from_numpy(get_mel(tgt_path)).float().unsqueeze(0)
    mel_target = mel_target.cuda()
    mel_target_lengths = torch.LongTensor([mel_target.shape[-1]])
    mel_target_lengths = mel_target_lengths.cuda()
    embed_target = torch.from_numpy(emb).float().unsqueeze(0)
    embed_target = embed_target.cuda()

    mel_source_tempo = get_mel(src_path)
    mel_source_tempo = (mel_source_tempo - mean[:, None]) / std[:, None]
    mel_source_tempo = torch.from_numpy(mel_source_tempo).float().unsqueeze(0)
    mel_source_tempo = mel_source_tempo.cuda()
    mel_source_lengths_tempo = torch.LongTensor([mel_source_tempo.shape[-1]])
    mel_source_lengths_tempo = mel_source_lengths_tempo.cuda()

    _, mel_modified = generator(mel_source_tempo, mel_source_lengths_tempo, mel_target, mel_target_lengths, embed_target,
                                          n_timesteps=100, mode='ml')

    mel_synth_np_modified = mel_modified.cpu().detach().squeeze().numpy()
    mel_synth_np_modified = (mel_synth_np_modified * std[:, None]) + mean[:, None]
    np.save(os.path.join(save_path, src_path.split('/')[-1].replace('.wav', '.npy')), mel_synth_np_modified)

def source_save(src_path, save_path):
    mel_source = get_mel(src_path).squeeze()
    np.save(save_path.replace('.wav', '.npy'), mel_source)

def get_avg_emb(emb_dir):
    allembs = []
    for root, dirs, files in os.walk(emb_dir):
        for f in files:
            if f.endswith('.npy'):
                allembs.append(np.load(os.path.join(root, f)))
    allembs = np.array(allembs)
    allembs = np.mean(allembs, 0)
    print(f'Embedding shape: {allembs.shape}')
    return allembs

def main(args, dys):
    stats = np.loadtxt(os.path.join(args.mean_std_file_ua, dys, 'global_mean_var.txt'))
    mean = stats[0]
    std = stats[1]
    stats_lt = np.loadtxt(os.path.join(args.mean_std_file, 'global_mean_var.txt'))
    mean_lt = stats_lt[0]
    std_lt = stats_lt[1]
    vc_path = os.path.join(args.model_path_dir, dys, 'vc.pt')
    vc_path_lt = args.model_path_dir_lt
    emb_dir = os.path.join(args.emb_dir, dys)
    # vocoder_path = os.path.join(args.vocoder_dir, dys)
    # vocoder_path_lt = args.vocoder_dir_lt
    results_dir = os.path.join(args.results_dir, dys)
    results_dir_lt = os.path.join(args.results_dir_lt, dys)
    results_dir_s = os.path.join(args.results_dir_s, dys)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(results_dir_lt, exist_ok=True)
    os.makedirs(results_dir_s, exist_ok=True)
    cmds = []
    target_cmds = []
    for root, dir, files in os.walk(args.gsc_dir):
        for f in files:
            if f.endswith('.wav'):
                cmds.append(os.path.join(root, f))
    print(len(cmds))
    for root, dir, files in os.walk(os.path.join(args.aty_dir, dys)):
        for f in files:
            if f.endswith('.wav'):
                target_cmds.append(os.path.join(root, f))
    print(len(target_cmds))
    random.shuffle(cmds)
    cmds = cmds[:5000]
    if args.debug:
        cmds = cmds[:2]
        target_cmds = target_cmds[:2]

    allembs = get_avg_emb(emb_dir)
    # loading voice conversion model
    generator = DiffVC(params.n_mels, params.channels, params.filters, params.heads,
                       params.layers, params.kernel, params.dropout, params.window_size,
                       params.enc_dim, params.spk_dim, params.use_ref_t, params.dec_dim,
                       params.beta_min, params.beta_max)
    generator.load_state_dict(torch.load(vc_path, map_location='cpu'))
    generator = generator.cuda()
    generator.eval()

    generator_lt = DiffVC(params.n_mels, params.channels, params.filters, params.heads,
                       params.layers, params.kernel, params.dropout, params.window_size,
                       params.enc_dim, params.spk_dim, params.use_ref_t, params.dec_dim,
                       params.beta_min, params.beta_max)
    generator_lt.load_state_dict(torch.load(vc_path_lt, map_location='cpu'))
    generator_lt = generator_lt.cuda()
    generator_lt.eval()

    for c in tqdm(cmds):
        random.shuffle(target_cmds)
        tgt_path = target_cmds[0]
        source_save(c, os.path.join(results_dir_s, c.split('/')[-1]))
        inference(generator, src_path=c, tgt_path=tgt_path, save_path=results_dir, mean=mean, std=std, emb=allembs)
        inference(generator_lt, src_path=c, tgt_path=tgt_path, save_path=results_dir_lt, mean=mean_lt, std=std_lt, emb=allembs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str,
                        default='./DuTaVC/results_dataug_fortts/aty')
    parser.add_argument('--results_dir_lt', type=str,
                        default='./DuTaVC/results_dataug_fortts/ty')
    parser.add_argument('--results_dir_s', type=str,
                        default='./DuTaVC/results_dataug_fortts/source')
    parser.add_argument('--model_path_dir', type=str,
                        default='./DuTaVC/logs_dec_aty')
    parser.add_argument('--model_path_dir_lt', type=str,
                        default='./DuTaVC/logs_dec_LT/vc.pt')
    parser.add_argument('--vocoder_dir', type=str,
                        default='./AtyTTS/checkpts/vocoder')
    parser.add_argument('--vocoder_dir_lt', type=str,
                        default='./AtyTTS/checkpts/vocoder/g_02500000')
    parser.add_argument('--aty_dir', type=str,
                        default='./DuTaVC/aty_data/wavs')
    parser.add_argument('--gsc_dir', type=str,
                        default='/data/lmorove1/hwang258/LibriTTS/LibriTTS_22050')
    parser.add_argument('--mean_std_file', type=str,
                        default='./DuTaVC/libritts_data/')
    parser.add_argument('--mean_std_file_ua', type=str,
                        default='./DuTaVC/aty_data/stats/')
    parser.add_argument('--emb_dir', type=str,
                        default='./DuTaVC/aty_data/embeds/')
    parser.add_argument('--dys', type=str, default='0005')
    parser.add_argument('--debug', action='store_true')
    parser.set_defaults(debug=False)
    args = parser.parse_args()
    main(args, args.dys)