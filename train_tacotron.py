import torch
from torch import optim
import torch.nn.functional as F
from utils import hparams as hp
from utils.display import *
from utils.dataset import get_tts_datasets
from utils.text.symbols import symbols
from utils.paths import Paths
from models.tacotron import Tacotron
import argparse
from utils import data_parallel_workaround
import os
from pathlib import Path
import time
import numpy as np
import sys
from utils.checkpoints import save_checkpoint, restore_checkpoint
from logger import Tacotron2Logger

import matplotlib
# matplotlib.use("MacOSX")

def np_now(x: torch.Tensor): return x.detach().cpu().numpy()

def prepare_directories_and_logger(output_directory, log_directory):
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    return logger

def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train Tacotron TTS')
    parser.add_argument('--force_train', '-f', action='store_true', help='Forces the model to train past total steps')
    parser.add_argument('--force_gta', '-g', action='store_true', help='Force the model to create GTA features')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
    args = parser.parse_args()

    hp.configure(args.hp_file)  # Load hparams from file
    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    force_train = args.force_train
    force_gta = args.force_gta

    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        for session in hp.tts_schedule:
            _, _, batch_size = session
            if batch_size % torch.cuda.device_count() != 0:
                raise ValueError('`batch_size` must be evenly divisible by n_gpus!')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    # Instantiate Tacotron Model
    print('\nInitialising Tacotron Model...\n')
    model = Tacotron(embed_dims=hp.tts_embed_dims,
                     num_chars=len(symbols),
                     encoder_dims=hp.tts_encoder_dims,
                     decoder_dims=hp.tts_decoder_dims,
                     decoder_R=hp.tts_R_train,
                     fft_bins=None,
                     postnet_dims=None,
                     encoder_K=hp.tts_encoder_K,
                     lstm_dims=hp.tts_lstm_dims,
                     postnet_K=None,
                     num_highways=hp.tts_num_highways,
                     dropout=hp.tts_dropout,
                     stop_threshold=hp.tts_stop_threshold).to(device)

    optimizer = optim.Adam(model.parameters())
    restore_checkpoint('tts', paths, model, optimizer, create_if_missing=True)

    scaler = torch.cuda.amp.GradScaler()

    # logger = prepare_directories_and_logger("/content/drive/MyDrive/Colab Notebooks/voiceclone/full_wave_tacotron_model/model_outputs/ljspeech_lsa_smooth_attention", "logdir")
    logger = prepare_directories_and_logger(paths.tts_output, "logdir")

    if not force_gta:
        for i, session in enumerate(hp.tts_schedule):
            current_step = model.get_step()

            lr, max_step, batch_size = session

            training_steps = max_step - current_step

            # Do we need to change to the next session?
            if current_step >= max_step:
                # Are there no further sessions than the current one?
                if i == len(hp.tts_schedule)-1:
                    # There are no more sessions. Check if we force training.
                    if force_train:
                        # Don't finish the loop - train forever
                        training_steps = 999_999_999
                    else:
                        # We have completed training. Breaking is same as continue
                        break
                else:
                    # There is a following session, go to it
                    continue

            model.r = hp.tts_R_train

            simple_table([(f'Steps with r={hp.tts_R_train}', str(training_steps//1000) + 'k Steps'),
                            ('Batch Size', batch_size),
                            ('Learning Rate', lr),
                            ('Outputs/Step (r)', model.r)])

            train_set, attn_example = get_tts_datasets(paths.data, batch_size, hp.tts_R_train)
            tts_train_loop(paths, model, scaler, logger, optimizer, train_set, lr, training_steps, attn_example)

        print('Training Complete.')
        print('To continue training increase tts_total_steps in hparams.py or use --force_train\n')


    # print('Creating Ground Truth Aligned Dataset...\n')

    # train_set, attn_example = get_tts_datasets(paths.data, 8, model.r)
    # create_gta_features(model, train_set, paths.gta)

    # print('\n\nYou can now train WaveRNN on GTA features - use python train_wavernn.py --gta\n')


def tts_train_loop(paths: Paths, model: Tacotron, scaler, logger, optimizer, train_set, lr, train_steps, attn_example):
    device = next(model.parameters()).device  # use same device as model parameters

    for g in optimizer.param_groups: g['lr'] = lr

    test_set, train_set = train_set

    duration = 0
    total_iters = len(train_set)
    epochs = train_steps // total_iters + 1

    for e in range(1, epochs+1):

        start = time.time()
        running_loss = 0

        # Perform 1 epoch
        for i, (x, wav, ids, _, stop_targets) in enumerate(train_set, 1):

            x, wav = x.to(device), wav.to(device)
            stop_targets = stop_targets.to(device)
            stop_targets.requires_grad = False

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
              # Parallelize model onto GPUS using workaround due to python bug
              if device.type == 'cuda' and torch.cuda.device_count() > 1:
                  logplists, logdetlosts, attention, stop_outputs = data_parallel_workaround(model, x, wav)
              else:
                  logplists, logdetlosts, attention, stop_outputs = model(x, wav)

              nll = -logplists - logdetlosts
              nll = nll / (wav.shape[2] / model.r) / wav.shape[1]
              nll = nll.mean()
              stop_loss = F.binary_cross_entropy_with_logits(stop_outputs, stop_targets)

              loss = nll + stop_loss

            scaler.scale(loss).backward()
            if hp.tts_clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.tts_clip_grad_norm)
                if grad_norm.isnan():
                    print('grad_norm was NaN!')

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            avg_loss = running_loss / i

            prev_duration = duration
            duration = (time.time() - start)
            speed = duration / i

            step = model.get_step()
            # k = step // 1000

            if step % hp.tts_checkpoint_every == 0 or step == 1:
                with torch.no_grad():
                    ckpt_name = f'taco_step{step}'
                    save_checkpoint('tts', paths, model, optimizer,
                                    name=ckpt_name, is_silent=True)
                    logger.log_training(loss.item(), grad_norm, lr, duration - prev_duration, step)

                    for k, (x_eval, wav_eval, ids_eval, _, stop_targets_eval) in enumerate(test_set, 1):
                        logplists_, logdetlosts_, attention_, stop_outputs_ = model(x_eval, wav_eval)

                        nll_ = -logplists_ - logdetlosts_
                        nll_ = nll_ / (wav_eval.shape[2] / model.r) / wav_eval.shape[1]
                        nll_ = nll_.mean()
                        stop_loss_ = F.binary_cross_entropy_with_logits(stop_outputs_, stop_targets_eval)

                        loss_ = nll_ + stop_loss_
                        break # validate for first 8 batchs 

                    # zlast, _, _, zlist = model.decoder.flows(wav[0, :, 0].view(1, 10//2, 96*2), model.decoder.step_zero_embbeding_features[0].unsqueeze(0))
                    # abc = model.decoder.flows.reverse([zlist[-1]], model.decoder.step_zero_embbeding_features[0].unsqueeze(0), reconstruct=True)
                    # print("Reverse flow wave and Groundtruth diff: ", (wav[0, :, 0] - abc[0]).mean())

                    logger.log_validation(loss_.item(), stop_targets_eval, [stop_outputs_, attention_], step)

            if attn_example in ids:
                idx = ids.index(attn_example)
                save_attention(np_now(attention[idx][:, :160]), paths.tts_attention/f'{step}')
                # save_spectrogram(np_now(m2_hat[idx]), paths.tts_mel_plot/f'{step}', 600)

            msg = f'|Epoch: {e}/{epochs} ({i}/{total_iters}) | Avg Loss: {avg_loss:#.4} | NLL: {nll.item():#.4} | StopLoss: {stop_loss.item():#.4} | {speed:#.2} s/iteration | Step: {step} | '
            print(msg)

        # Must save latest optimizer state to ensure that resuming training
        # doesn't produce artifacts
        save_checkpoint('tts', paths, model, optimizer, is_silent=True)
        model.log(paths.tts_log, msg)
        print(' ')


def create_gta_features(model: Tacotron, train_set, save_path: Path):
    device = next(model.parameters()).device  # use same device as model parameters

    iters = len(train_set)

    for i, (x, mels, ids, mel_lens) in enumerate(train_set, 1):

        x, mels = x.to(device), mels.to(device)

        with torch.no_grad(): _, gta, _ = model(x, mels)

        gta = gta.cpu().numpy()

        for j, item_id in enumerate(ids):
            mel = gta[j][:, :mel_lens[j]]
            mel = (mel + 4) / 8
            np.save(save_path/f'{item_id}.npy', mel, allow_pickle=False)

        bar = progbar(i, iters)
        msg = f'{bar} {i}/{iters} Batches '
        stream(msg)


if __name__ == "__main__":
    main()
