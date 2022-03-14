import pickle
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from utils.dsp import *
from utils import hparams as hp
from utils.text import text_to_sequence
from utils.paths import Paths
from pathlib import Path


###################################################################################
# WaveRNN/Vocoder Dataset #########################################################
###################################################################################


class VocoderDataset(Dataset):
    def __init__(self, path: Path, dataset_ids, train_gta=False):
        self.metadata = dataset_ids
        self.mel_path = path/'gta' if train_gta else path/'mel'
        self.quant_path = path/'quant'


    def __getitem__(self, index):
        item_id = self.metadata[index]
        m = np.load(self.mel_path/f'{item_id}.npy')
        x = np.load(self.quant_path/f'{item_id}.npy')
        return m, x

    def __len__(self):
        return len(self.metadata)


def get_vocoder_datasets(path: Path, batch_size, train_gta):

    with open(path/'dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    dataset_ids = [x[0] for x in dataset]

    random.seed(1234)
    random.shuffle(dataset_ids)

    test_ids = dataset_ids[-hp.voc_test_samples:]
    train_ids = dataset_ids[:-hp.voc_test_samples]

    train_dataset = VocoderDataset(path, train_ids, train_gta)
    test_dataset = VocoderDataset(path, test_ids, train_gta)

    train_set = DataLoader(train_dataset,
                           collate_fn=collate_vocoder,
                           batch_size=batch_size,
                           num_workers=2,
                           shuffle=True,
                           pin_memory=True)

    test_set = DataLoader(test_dataset,
                          batch_size=1,
                          num_workers=1,
                          shuffle=False,
                          pin_memory=True)

    return train_set, test_set


def collate_vocoder(batch):
    mel_win = hp.voc_seq_len // hp.hop_length + 2 * hp.voc_pad
    max_offsets = [x[0].shape[-1] -2 - (mel_win + 2 * hp.voc_pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + hp.voc_pad) * hp.hop_length for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]

    labels = [x[1][sig_offsets[i]:sig_offsets[i] + hp.voc_seq_len + 1] for i, x in enumerate(batch)]

    mels = np.stack(mels).astype(np.float32)
    labels = np.stack(labels).astype(np.int64)

    mels = torch.tensor(mels)
    labels = torch.tensor(labels).long()

    x = labels[:, :hp.voc_seq_len]
    y = labels[:, 1:]

    bits = 16 if hp.voc_mode == 'MOL' else hp.bits

    x = label_2_float(x.float(), bits)

    if hp.voc_mode == 'MOL':
        y = label_2_float(y.float(), bits)

    return x, y, mels


###################################################################################
# Tacotron/TTS Dataset ############################################################
###################################################################################

from os import path as ppp

def get_tts_datasets(path: Path, batch_size, r):

    with open(path/'dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    dataset_ids = []
    wav_lengths = []

    for (item_id, length) in dataset:
        if not ppp.exists(path/'norm_wav'/f'{item_id}.npy'): continue

        if length <= hp.tts_max_wav_len:
            dataset_ids += [item_id]
            wav_lengths += [length]

    with open(path/'text_dict.pkl', 'rb') as f:
        text_dict = pickle.load(f)

    num_all_datasets = len(dataset_ids)

    train_ratio = 0.99
    train_dataset = TTSDataset(path, dataset_ids[:int(train_ratio*num_all_datasets)], text_dict)
    test_dataset = TTSDataset(path, dataset_ids[int(train_ratio*num_all_datasets):], text_dict)

    sampler = None

    if hp.tts_bin_lengths:
        sampler = BinnedLengthSampler(wav_lengths, batch_size, batch_size * 3)

    train_set = DataLoader(train_dataset,
                           collate_fn=lambda batch: collate_tts(batch, r),
                           batch_size=batch_size,
                           sampler=sampler,
                           num_workers=0,
                           pin_memory=True)

    test_set = DataLoader(test_dataset,
                           collate_fn=lambda batch: collate_tts(batch, r),
                           batch_size=8,
                           shuffle=True,
                           sampler=sampler,
                           num_workers=0,
                           pin_memory=True)

    longest = wav_lengths.index(max(wav_lengths))

    # Used to evaluate attention during training process
    attn_example = dataset_ids[longest]

    # print(attn_example)

    return [test_set, train_set], attn_example


class TTSDataset(Dataset):
    def __init__(self, path: Path, dataset_ids, text_dict):
        self.path = path
        self.metadata = dataset_ids
        self.text_dict = text_dict

    def __getitem__(self, index):
        item_id = self.metadata[index]
        x = text_to_sequence(self.text_dict[item_id], hp.tts_cleaner_names)
        wav = np.load(self.path/'norm_wav'/f'{item_id}.npy')
        wav_len = wav.shape[-1]
        return x, wav, item_id, wav_len

    def __len__(self):
        return len(self.metadata)


def pad1d(x, max_len):
    return np.pad(x, (0, max_len - len(x)), mode='constant')


def pad2d(x, max_len):
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode='constant')


def collate_tts(batch, r):

    x_lens = [len(x[0]) for x in batch]
    max_x_len = max(x_lens)

    chars = [pad1d(x[0], max_x_len) for x in batch]
    chars = np.stack(chars)

    wav_split_r = [torch.from_numpy(pre_emphasis(x[1]).astype("float32")).unfold(0, hp.tts_K(r), hp.tts_K(r)).T for x in batch]

    spec_lens = [x.shape[-1] for x in wav_split_r]
    max_spec_len = max(spec_lens) + 1
    if max_spec_len % r != 0:
        max_spec_len += r - max_spec_len % r

    wav = [pad2d(x, max_spec_len) for x in wav_split_r]
    wav = np.stack(wav)

    ids = [x[2] for x in batch]
    wav_lens = [x[3] for x in batch]

    chars = torch.tensor(chars).long()
    wav = torch.tensor(wav)

    stop_targets = torch.ones(len(batch), max_spec_len)
    for j in range(len(ids)):
        stop_targets[j, :wav_split_r[j].size(1)-1] = 0

    # scale spectrograms to -4 <--> 4
    # mel = (mel * 8.) - 4.
    return chars, wav, ids, wav_lens, stop_targets


class BinnedLengthSampler(Sampler):
    def __init__(self, lengths, batch_size, bin_size):
        _, self.idx = torch.sort(torch.tensor(lengths).long())
        self.batch_size = batch_size
        self.bin_size = bin_size
        assert self.bin_size % self.batch_size == 0

    def __iter__(self):
        # Need to change to numpy since there's a bug in random.shuffle(tensor)
        # TODO: Post an issue on pytorch repo
        idx = self.idx.numpy()
        bins = []

        for i in range(len(idx) // self.bin_size):
            this_bin = idx[i * self.bin_size:(i + 1) * self.bin_size]
            random.shuffle(this_bin)
            bins += [this_bin]

        random.shuffle(bins)
        binned_idx = np.stack(bins).reshape(-1)

        if len(binned_idx) < len(idx):
            last_bin = idx[len(binned_idx):]
            random.shuffle(last_bin)
            binned_idx = np.concatenate([binned_idx, last_bin])

        return iter(torch.tensor(binned_idx).long())

    def __len__(self):
        return len(self.idx)
