# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
import argparse
import numpy as np
from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import params
from model import GradTTS
from data import TextMelAugDataset, TextMelAugBatchCollate
from utils import plot_tensor, save_plot
from text.symbols import symbols

data_path = params.data_path
cmudict_path = params.cmudict_path
add_blank = params.add_blank

log_dir_aug = params.log_dir_aug
n_epochs = params.n_epochs
batch_size = params.batch_size
out_size = params.out_size
learning_rate = params.learning_rate
random_seed = params.seed

nsymbols = len(symbols) + 1 if add_blank else len(symbols)
n_enc_channels = params.n_enc_channels
filter_channels = params.filter_channels
filter_channels_dp = params.filter_channels_dp
n_enc_layers = params.n_enc_layers
enc_kernel = params.enc_kernel
enc_dropout = params.enc_dropout
n_heads = params.n_heads
window_size = params.window_size

n_feats = params.n_feats
n_fft = params.n_fft
sample_rate = params.sample_rate
hop_length = params.hop_length
win_length = params.win_length
f_min = params.f_min
f_max = params.f_max

dec_dim = params.dec_dim
beta_min = params.beta_min
beta_max = params.beta_max
pe_scale = params.pe_scale
pretrained_path = params.pretrained_path

def main(train_filelist_path, valid_filelist_path, log_path):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print('Initializing logger...')
    logger = SummaryWriter(log_dir=log_path)

    print('Initializing data loaders...')
    train_dataset = TextMelAugDataset(train_filelist_path, cmudict_path, add_blank,
                                   n_fft, n_feats, sample_rate, hop_length,
                                   win_length, f_min, f_max)
    batch_collate = TextMelAugBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=4, shuffle=False)
    test_dataset = TextMelAugDataset(valid_filelist_path, cmudict_path, add_blank,
                                  n_fft, n_feats, sample_rate, hop_length,
                                  win_length, f_min, f_max)

    print('Initializing model...')
    model = GradTTS(nsymbols, 1, None, n_enc_channels, filter_channels, filter_channels_dp, 
                    n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                    n_feats, dec_dim, beta_min, beta_max, pe_scale).cuda()
    model.load_state_dict(torch.load(pretrained_path))
    print('Number of encoder + duration predictor parameters: %.2fm' % (model.encoder.nparams/1e6))
    print('Number of decoder parameters: %.2fm' % (model.decoder.nparams/1e6))
    print('Total parameters: %.2fm' % (model.nparams/1e6))

    print('Initializing optimizer...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    print('Logging test batch...')
    test_batch = test_dataset.sample_test_batch(size=params.test_size)
    for i, item in enumerate(test_batch):
        mel = item['y'] # target mel-spectrogram
        mel_aty = item['aty']  # vc-generated atypical mel-spectrogram
        mel_s = item['s']  # source typical mel-spectrogram for vc
        mel_ty = item['ty'] # vc-generated typical mel-spectrogram
        logger.add_image(f'image_{i}/aty', plot_tensor(mel_aty.squeeze()),
                         global_step=0, dataformats='HWC')
        logger.add_image(f'image_{i}/ty', plot_tensor(mel_ty.squeeze()),
                         global_step=0, dataformats='HWC')
        logger.add_image(f'image_{i}/source', plot_tensor(mel_s.squeeze()),
                    global_step=0, dataformats='HWC')
        logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),
                         global_step=0, dataformats='HWC')
        save_plot(mel.squeeze(), f'{log_path}/original_{i}.png')
        save_plot(mel_aty.squeeze(), f'{log_path}/aty_{i}.png')
        save_plot(mel_ty.squeeze(), f'{log_path}/ty_{i}.png')
        save_plot(mel_s.squeeze(), f'{log_path}/source_{i}.png')

    print('Start training...')
    iteration = 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        dur_losses = []
        prior_losses = []
        diff_losses = []
        spk_losses = []
        aty_losses = []
        with tqdm(loader, total=len(train_dataset)//batch_size) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                model.zero_grad()
                x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
                y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
                aty, ty, s, aty_lengths = batch['aty'].cuda(), batch['ty'].cuda(), batch['s'].cuda(), batch['aty_lengths'].cuda()
                # x: batch of texts, converted to a tensor with phoneme embedding ids.
                # x_lengths: lengths of texts in batch.
                # y: batch of corresponding mel-spectrograms.
                # y_lengths: lengths of mel-spectrograms in batch
                # aty: batch of vc-generated atypical mel-spectrograms
                # ty: batch of vc-generated typical mel-spectrograms
                # s: batch of source typical mel-spectrograms for vc
                # aty_lengths: lengths of vc-generated atypical mel-spectrograms in batch
                dur_loss, prior_loss, diff_loss = model.compute_loss(x, x_lengths,
                                                                     y, y_lengths,
                                                                     out_size=out_size)
                spk_loss = model.compute_loss_aug(s, aty, aty_lengths, out_size=out_size)
                aty_loss = model.compute_loss_aug(ty, aty, aty_lengths, out_size=out_size)
                loss = sum([dur_loss, prior_loss, diff_loss, spk_loss, aty_loss])
                loss.backward()

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(),
                                                               max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),
                                                               max_norm=1)
                optimizer.step()

                logger.add_scalar('training/duration_loss', dur_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/prior_loss', prior_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/diffusion_loss', diff_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/speaker_loss', spk_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/atypical_loss', aty_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                  global_step=iteration)
                logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                  global_step=iteration)
                
                dur_losses.append(dur_loss.item())
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())
                spk_losses.append(spk_loss.item())
                aty_losses.append(aty_loss.item())

                msg = f'Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}, aty_loss: {aty_loss.item()}, spk_loss: {spk_loss.item()}'
                progress_bar.set_description(msg)
                
                iteration += 1

        log_msg = 'Epoch %d: duration loss = %.3f ' % (epoch, np.mean(dur_losses))
        log_msg += '| prior loss = %.3f ' % np.mean(prior_losses)
        log_msg += '| diffusion loss = %.3f' % np.mean(diff_losses)
        log_msg += '| speaker loss = %.3f' % np.mean(spk_losses)
        log_msg += '| atypical loss = %.3f\n' % np.mean(aty_losses)
        with open(f'{log_path}/train.log', 'a') as f:
            f.write(log_msg)

        if epoch % params.save_every > 0:
            continue

        model.eval()
        print('Synthesis...')
        with torch.no_grad():
            for i, item in enumerate(test_batch):
                x = item['x'].to(torch.long).unsqueeze(0).cuda()
                aty = item['aty'].unsqueeze(0).cuda()
                ty = item['ty'].unsqueeze(0).cuda()
                s = item['s'].unsqueeze(0).cuda()
                aty_lengths = torch.LongTensor([aty.shape[-1]]).cuda()
                x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                y_enc, y_dec, attn = model(x, x_lengths, n_timesteps=50)
                ty_dec = model.forward_aug(ty, aty_lengths, n_timesteps=50)
                s_dec = model.forward_aug(s, aty_lengths, n_timesteps=50)
                logger.add_image(f'image_{i}/generated_enc',
                                 plot_tensor(y_enc.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/generated_dec',
                                 plot_tensor(y_dec.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/ty_dec',
                                 plot_tensor(ty_dec.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/s_dec',
                                 plot_tensor(s_dec.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/alignment',
                                 plot_tensor(attn.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                save_plot(y_enc.squeeze().cpu(), 
                          f'{log_path}/generated_enc_{i}.png')
                save_plot(y_dec.squeeze().cpu(), 
                          f'{log_path}/generated_dec_{i}.png')
                save_plot(attn.squeeze().cpu(), 
                          f'{log_path}/alignment_{i}.png')
                save_plot(ty_dec.squeeze().cpu(), 
                          f'{log_path}/ty_dec_{i}.png')
                save_plot(s_dec.squeeze().cpu(), 
                          f'{log_path}/s_dec_{i}.png')
                

        ckpt = model.state_dict()
        torch.save(ckpt, f=f"{log_path}/last.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--spk', type=str, default=None)
    args = parser.parse_args()
    spk = args.spk
    train_filelist_path = os.path.join(data_path, spk, 'train_aug.txt')
    valid_filelist_path = os.path.join(data_path, spk, 'test_aug.txt')
    log_path = os.path.join(log_dir_aug, spk, 'ep'+str(n_epochs))
    main(train_filelist_path, valid_filelist_path, log_path)
