import socket
import sys
import os

import math
import time
import glob
import datetime
import random
import pickle
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from numba import jit
from pyOSC3 import OSC3
from torch.nn import DataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.builders import RecurrentEncoderBuilder
from fast_transformers.masking import TriangularCausalMask
# from fast_transformers.masking import FullMask

import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note

import saver

from MIDIBERT.MidiBERTEmotion import MidiBERTEmotion
# from modules import ProjectedAdaptiveLogSoftmax, AdaptiveEmbedding, gumbel_softmax, gumbel_softmax_sample, sample_gumbel, ArgMax
from default import process_data_npy, process_kinds_data

################################################################################
# config
################################################################################

# MODE = 'train'
# MODE = 'inference'
MODE = 'save_npy'

###--- data ---###
path_data_root = '../../../dataset/MIDI-BERT-Emotion/emopia-CP'
path_train_data = path_data_root + '/train_data_linear.npz'
path_dictionary = path_data_root + '/CP_dict_new.pkl'

###--- answer data ---###
path_cls_answer_data = path_data_root + '/emopia_cp_train_ans.npy'

###--- training config ---###
D_MODEL = 512
N_LAYER = 12
N_HEAD = 8
path_exp = 'only_nll'
batch_size = 8
gid = 0
init_lr = 0.0001

###--- fine-tuning & inference config ---###
info_load_model = (
    # path to ckpt for loading
    'only_nll',
    # loss
    'high'
)
# info_load_model = None                  

path_gendir = './only_nll/midis'
per_class_num_songs = 100

################################################################################
# File IO
################################################################################

BEAT_RESOL = 480
# BEAT_RESOL = BEAT_RESOLUTION
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4        # position 每个的间隔


def write_midi(words, path_outfile, word2event):
    class_keys = word2event.keys()
    midi_obj = miditoolkit.midi.parser.MidiFile()

    bar_cnt = 0
    cur_pos = 0

    all_notes = []

    cnt_error = 0
    for i in range(len(words)):
        vals = []
        for kidx, key in enumerate(class_keys):
            vals.append(word2event[key][words[i][kidx]])
        # print(vals)

        if vals[0].split(' ')[-1] == 'New':
            bar_cnt += 1
        if (vals[0].split(' ')[-1] == 'New') or (vals[0].split(' ')[-1] == 'Continue'):
            position_ = vals[1].split(' ')[-1]
            if position_ != "<PAD>" and position_ != "<MASK>":
                position = int(position_.split('/')[0]) - 1
            else:
                continue
            # current_bar_st = bar_cnt * BAR_RESOL
            # current_bar_et = (bar_cnt + 1) * BAR_RESOL
            # flags = np.linspace(current_bar_st, current_bar_et, BEAT_RESOL, endpoint=False, dtype=int)
            # st = flags[position]
            duration_ = vals[3].split(' ')[-1]
            if duration_ != "<PAD>" and duration_ != "<MASK>":
                duration = int(duration_) * 60
            else:
                continue
            pitch_ = vals[2].split(' ')[-1]
            if pitch_ != "<PAD>" and pitch_ != "<MASK>":
                pitch = int(pitch_)
            else:
                continue
            st = bar_cnt * BAR_RESOL + position * TICK_RESOL
            et = st + duration
            all_notes.append(miditoolkit.Note(velocity=60, pitch=pitch, start=st, end=et))

    # save midi
    piano_track = Instrument(0, is_drum=False, name='piano')
    piano_track.notes = all_notes
    midi_obj.instruments = [piano_track]
    midi_obj.dump(path_outfile)


def send_udp(ip, port):
    c = OSC3.OSCClient()
    c.connect((ip, port))
    oscmsg = OSC3.OSCMessage()
    oscmsg.setAddress("/play")
    oscmsg.append('HELLO')
    c.send(oscmsg)


################################################################################
# Sampling
################################################################################
# -- temperature -- #
@jit(nopython=True)
def softmax_with_temperature(logits, temperature):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    return probs


def weighted_sampling(probs):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    word = np.random.choice(sorted_index, size=1, p=sorted_probs)[0]
    return word


# -- nucleus -- #
def nucleus(probs, p):
    probs /= (sum(probs) + 1e-5)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    if np.NaN in candi_probs:
        print("NaN appeared in candidate probabilities, returning None word.")
        return None
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word


def sampling(logit, p=None, t=1.0, is_tensor=False):
    if is_tensor:
        logit = logit.squeeze().detach().cpu().numpy()
    else:
        logit = logit.squeeze().cpu().numpy()
    probs = softmax_with_temperature(logits=logit, temperature=t)

    if p is not None:
        cur_word = nucleus(probs, p=p)
        if cur_word == None:
            cur_word = weighted_sampling(probs)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word


################################################################################
# Model
################################################################################


def network_paras(model):
    # compute only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, n_token, is_training=True):
        super(TransformerModel, self).__init__()

        # --- params config --- #
        self.n_token = n_token
        # embed length
        self.d_model = D_MODEL
        self.n_layer = N_LAYER  #
        self.dropout = 0.1
        self.n_head = N_HEAD  #
        self.d_head = D_MODEL // N_HEAD
        self.d_inner = 2048
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.emb_sizes = [32, 128, 256, 256, 32, 64]

        # --- modules config --- #
        # embeddings
        print('>>>>>:', self.n_token)
        self.word_emb_bar = Embeddings(self.n_token[0], self.emb_sizes[0])
        self.word_emb_position = Embeddings(self.n_token[1], self.emb_sizes[1])
        self.word_emb_pitch = Embeddings(self.n_token[2], self.emb_sizes[2])
        self.word_emb_duration = Embeddings(self.n_token[3], self.emb_sizes[3])
        self.word_emb_type = Embeddings(self.n_token[4], self.emb_sizes[4])
        self.word_emb_emotion = Embeddings(self.n_token[5], self.emb_sizes[5])
        # position == d_model
        self.pos_emb = PositionalEncoding(self.d_model, self.dropout)
        self.is_training = is_training

        # linear
        self.in_linear = nn.Linear(np.sum(self.emb_sizes, dtype=np.int).item(), self.d_model)

        # encoder
        if is_training:
            # encoder (training)
            self.transformer_encoder = TransformerEncoderBuilder.from_kwargs(
                n_layers=self.n_layer,
                n_heads=self.n_head,
                query_dimensions=self.d_model // self.n_head,
                value_dimensions=self.d_model // self.n_head,
                feed_forward_dimensions=2048,
                activation='gelu',
                dropout=0.1,
                attention_type="causal-linear",
            ).get()
        else:
            # encoder (inference)
            print(' [o] using RNN backend.')
            self.transformer_encoder = RecurrentEncoderBuilder.from_kwargs(
                n_layers=self.n_layer,
                n_heads=self.n_head,
                query_dimensions=self.d_model // self.n_head,
                value_dimensions=self.d_model // self.n_head,
                feed_forward_dimensions=2048,
                activation='gelu',
                dropout=0.1,
                attention_type="causal-linear",
            ).get()

        # blend with type
        self.project_concat_type = nn.Linear(self.d_model + 64, self.d_model)

        # ProjectedAdaptiveLogSoftmax
        self.proj_bar = nn.Linear(self.d_model, self.n_token[0])
        self.proj_position = nn.Linear(self.d_model, self.n_token[1])
        self.proj_pitch = nn.Linear(self.d_model, self.n_token[2])
        self.proj_duration = nn.Linear(self.d_model, self.n_token[3])
        self.proj_type = nn.Linear(self.d_model, self.n_token[4])
        self.proj_emotion = nn.Linear(self.d_model, self.n_token[5])

    def forward(self, x):
        ret = self.forward_hidden(x, is_training=False)
        return ret

    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def loss_sum(self, loss):
        return torch.sum(loss) / loss.size(0)

    # nll loss
    def train_step(self, x, target, loss_mask):
        h, y_emotion = self.forward_hidden(x)
        y_bar, y_position, y_pitch, y_duration, y_type = self.forward_output(h, target)

        # reshape (b, s, f) -> (b, f, s)
        y_bar = y_bar[:, ...].permute(0, 2, 1)
        y_position = y_position[:, ...].permute(0, 2, 1)
        y_pitch = y_pitch[:, ...].permute(0, 2, 1)
        y_duration = y_duration[:, ...].permute(0, 2, 1)
        y_type = y_type[:, ...].permute(0, 2, 1)
        y_emotion = y_emotion[:, ...].permute(0, 2, 1)


        # loss
        loss_bar = self.compute_loss(
            y_bar, target[..., 0], loss_mask)
        loss_position = self.compute_loss(
            y_position, target[..., 1], loss_mask)
        loss_pitch = self.compute_loss(
            y_pitch, target[..., 2], loss_mask)
        loss_duration = self.compute_loss(
            y_duration, target[..., 3], loss_mask)
        loss_type = self.compute_loss(
            y_type, target[..., 4], loss_mask)
        loss_emotion = self.compute_loss(
            y_emotion, target[..., 5], loss_mask)

        return loss_bar, loss_position, loss_pitch, loss_duration, loss_type, loss_emotion

    # logits ==> gumbel_out
    def gumbel(self, logits, temperature=1):
        gumbel_sample = torch.distributions.gumbel.Gumbel(loc=torch.zeros_like(logits), scale=torch.ones_like(logits)).sample()
        gumbel_sample = torch.autograd.Variable(gumbel_sample)
        y = F.softmax((logits + gumbel_sample) / temperature, dim=-1)
        y_onehot = F.one_hot(torch.argmax(y, dim=-1), num_classes=y.shape[-1])
        return (y_onehot - y).detach() + y

    def forward_hidden(self, x, memory=None, is_training=True):
        '''
        linear transformer: b x s x f
        x.shape=(bs, nf)
        '''

        # embeddings
        emb_bar = self.word_emb_bar(x[..., 0])
        emb_position = self.word_emb_position(x[..., 1])
        emb_pitch = self.word_emb_pitch(x[..., 2])
        emb_duration = self.word_emb_duration(x[..., 3])
        emb_type = self.word_emb_type(x[..., 4])
        emb_emotion = self.word_emb_emotion(x[..., 5])

        embs = torch.cat(
            [
                emb_bar,
                emb_position,
                emb_pitch,
                emb_duration,
                emb_type,
                emb_emotion
            ], dim=-1)

        emb_linear = self.in_linear(embs)
        pos_emb = self.pos_emb(emb_linear)

        # assert False

        # transformer
        if is_training:
            # mask
            attn_mask = TriangularCausalMask(pos_emb.size(1), device=x.device)
            h = self.transformer_encoder(pos_emb, attn_mask)  # y: b x s x d_model

            y_emotion = self.proj_emotion(h)
            return h, y_emotion
        else:
            pos_emb = pos_emb.squeeze(0)
            h, memory = self.transformer_encoder(pos_emb, memory=memory)  # y: s x d_model

            # project type
            y_emotion = self.proj_emotion(h)
            return h, y_emotion, memory

    def forward_output(self, h, y):
        '''
        for training
        '''
        tf_skip_emotion = self.word_emb_emotion(y[..., 5])

        # project other
        y_concat_type = torch.cat([h, tf_skip_emotion], dim=-1)
        y_ = self.project_concat_type(y_concat_type)

        y_bar = self.proj_bar(y_)
        y_position = self.proj_position(y_)
        y_pitch = self.proj_pitch(y_)
        y_duration = self.proj_duration(y_)
        y_type = self.proj_type(y_)

        return y_bar, y_position, y_pitch, y_duration, y_type

    # for inference
    def froward_output_sampling(self, h, y_emotion):
        '''
            for inference
        '''
        cur_word_emotion = F.gumbel_softmax(y_emotion, tau=1, hard=True)
        cur_word_emotion_ = torch.argmax(cur_word_emotion, dim=-1)

        emotion_word_t = cur_word_emotion_.unsqueeze(0)

        tf_skip_emotion = self.word_emb_emotion(emotion_word_t).squeeze(0)        # tf_skip_composer.size: torch.Size([1, 64])

        # concat
        y_concat_type = torch.cat([h, tf_skip_emotion], dim=-1)
        y_ = self.project_concat_type(y_concat_type)

        # project other
        y_bar = self.proj_bar(y_)
        y_position = self.proj_position(y_)
        y_pitch = self.proj_pitch(y_)
        y_duration = self.proj_duration(y_)
        y_type = self.proj_type(y_)

        # sampling gen_cond                              

        # collect
        next_arr = self.get_current_gumbel_softmax(y_bar, y_position, y_pitch, y_duration, y_type, cur_word_emotion_)

        return next_arr

    def get_current_gumbel_softmax(self, y_bar, y_position, y_pitch, y_duration, y_type, next_emotion):
        y_next_bars = F.gumbel_softmax(y_bar, tau=1, hard=True)
        y_next_positions = F.gumbel_softmax(y_position, tau=1, hard=True)
        y_next_pitchs = F.gumbel_softmax(y_pitch, tau=1, hard=True)
        y_next_durations = F.gumbel_softmax(y_duration, tau=1, hard=True)
        y_next_type = F.gumbel_softmax(y_type, tau=1, hard=True)

        next_bars, next_positions, next_pitchs, next_durations, next_types = torch.argmax(y_next_bars, dim=-1), \
                                                                 torch.argmax(y_next_positions, dim=-1), \
                                                                 torch.argmax(y_next_pitchs, dim=-1), \
                                                                 torch.argmax(y_next_durations, dim=-1), \
                                                                torch.argmax(y_next_type, dim=-1)


        fake_nexts = torch.cat([torch.unsqueeze(next_bars, -1), torch.unsqueeze(next_positions, -1), torch.unsqueeze(next_pitchs, -1), torch.unsqueeze(next_durations, -1), torch.unsqueeze(next_types, -1), torch.unsqueeze(next_emotion, -1)], dim=-1)
        fake_nexts = fake_nexts.squeeze().cpu().numpy()

        return fake_nexts

    def print_word_cp(self, cp, classes, word2event):
        result = [word2event[k][cp[idx]] for idx, k in enumerate(classes)]

        for r in result:
            print('{:15s}'.format(str(r)), end=' | ')
        print('')

    def inference_from_scratch(self, dictionary, emotion_kind):
        event2word, word2event = dictionary

        init = np.array([
            [2, 16, 86, 64, 1, emotion_kind],  # set style， ['Bar <PAD>', 'Position <PAD>', 'Pitch <PAD>', 'Duration <PAD>', 1, 0||1||2||3]          #
        ])

        with torch.no_grad():
            final_res = []
            memory = None
            h = None

            init_t = torch.from_numpy(init).cuda()
            print('------ initiate ------')
            for step in range(init.shape[0]):
                # self.print_word_cp(init[step, :], classes, word2event)
                # print('init_t.size: ', init_t.size())                       # init_t.size:  torch.Size([1, 6])
                input_ = init_t[step, :].unsqueeze(0).unsqueeze(0)
                # print('input_.size: ', input_.size())                       # input_.size:  torch.Size([1, 1, 6])
                final_res.append(init[step, :][None, ...])

                h, y_emotion, memory = self.forward_hidden(
                    input_, memory, is_training=False)

            print('------ generate ------')
            # seq length
            for num_note in range(512):
                # sample others
                next_arr = self.froward_output_sampling(h, y_emotion)
                # print('next_arr.shape: ', next_arr.shape)                   # next_arr.shape:  (4,)
                final_res.append(next_arr[None, ...])
                # self.print_word_cp(next_arr, classes, word2event)

                # forward
                input_ = torch.from_numpy(next_arr).cuda()
                input_ = input_.unsqueeze(0).unsqueeze(0)
                h, y_emotion, memory = self.forward_hidden(
                    input_, memory, is_training=False)

        print('\n--------[Done]--------')
        final_res = np.concatenate(final_res)
        print(final_res.shape)
        return final_res



##########################################################################################################################
# Script
##########################################################################################################################


def train():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # hyper params
    n_epoch = 800
    max_grad_norm = 3

    # load
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary
    train_data = np.load(path_train_data)
    cls_answer_data = np.load(path_cls_answer_data, allow_pickle=True)

    # create saver
    saver_agent = saver.Saver(path_exp)

    # config
    n_class = []
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))

    # log
    print('num of classes:', n_class)

    # init
    net = TransformerModel(n_class)
    n_parameters = network_paras(net)
    print('n_parameters: {:,}'.format(n_parameters))
    saver_agent.add_summary_msg(
        ' > params amount: {:,d}'.format(n_parameters))

    # load model
    if info_load_model:
        path_ckpt = info_load_model[0]  # path to ckpt dir
        loss = info_load_model[1]  # loss
        name = 'loss_' + str(loss)
        path_saved_ckpt = os.path.join(path_ckpt, name + '_params.pt')
        print('[*] load model from:', path_saved_ckpt)
        # net.load_state_dict(torch.load(path_saved_ckpt, map_location=torch.device('cpu')))
        net.load_state_dict(torch.load(path_saved_ckpt))

    net = net.cuda()
    net.train()

    # optimizers
    optimizer = optim.Adam(net.parameters(), lr=init_lr)

    # unpack
    train_x = train_data['x']
    train_y = train_data['y']
    train_mask = train_data['mask']
    num_batch = len(train_x) // batch_size

    print('    num_batch:', num_batch)
    print('    train_x:', train_x.shape)
    print('    train_y:', train_y.shape)
    print('    train_mask:', train_mask.shape)
    print('    cls_answer:', cls_answer_data.shape)

    # run
    start_time = time.time()
    for epoch in range(n_epoch):
        acc_loss = 0
        cls_loss_ = 0
        g_loss_ = 0
        acc_losses = np.zeros(6)

        for bidx in range(num_batch):  # num_batch
            saver_agent.global_step_increment()

            # index
            bidx_st = batch_size * bidx
            bidx_ed = batch_size * (bidx + 1)

            # unpack batch data
            batch_x = train_x[bidx_st:bidx_ed]
            batch_y = train_y[bidx_st:bidx_ed]
            batch_mask = train_mask[bidx_st:bidx_ed]
            batch_answer = cls_answer_data[bidx_st:bidx_ed]

            # to tensor
            batch_x = torch.from_numpy(batch_x).cuda()
            batch_y = torch.from_numpy(batch_y).cuda()
            batch_mask = torch.from_numpy(batch_mask).cuda()
            batch_answer = torch.from_numpy(batch_answer).cuda()

            # run
            loss_bar, loss_position, loss_pitch, loss_duration, loss_type, loss_emotion \
                = net.train_step(batch_x, batch_y, batch_mask)
            nll_loss_arr = [loss_bar, loss_position, loss_pitch, loss_duration, loss_type, loss_emotion]
            nll_loss = (loss_bar + loss_position + loss_pitch + loss_duration + loss_type + loss_emotion) / 6

            cls_loss = 0
            # print(cls_loss)
            g_loss = nll_loss
            optimizer.zero_grad()
            g_loss.backward()
            if max_grad_norm is not None:
                clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()


            # acc
            acc_losses += np.array([l.item() for l in nll_loss_arr])
            acc_loss += nll_loss.item()
            cls_loss_ += cls_loss
            g_loss_ += g_loss.item()

            # log
            # saver_agent.add_summary('batch loss', nll_loss.item())

        # epoch loss
        runtime = time.time() - start_time
        epoch_loss = acc_loss / num_batch
        acc_losses = acc_losses / num_batch
        g_loss_ = g_loss_ / num_batch
        print('------------------------------------')
        print('epoch: {}/{} | Loss: {} | time: {}'.format(
            epoch, n_epoch, epoch_loss, str(datetime.timedelta(seconds=runtime))))
        each_loss_str = '{:04f}, {:04f}, {:04f}, {:04f}\r'.format(
            acc_losses[0], acc_losses[1], acc_losses[2], acc_losses[3])
        print('    >', each_loss_str)

        saver_agent.add_summary('*', '*'*25)
        saver_agent.add_summary('epoch loss', epoch_loss)
        saver_agent.add_summary('epoch each loss', each_loss_str)
        saver_agent.add_summary('epoch g loss', g_loss_)

        saver_agent.save_model(net, name='loss_high')

def generate():
    # path
    path_ckpt = info_load_model[0]  # path to ckpt dir
    loss = info_load_model[1]  # loss
    name = 'loss_' + str(loss)
    path_saved_ckpt = os.path.join(path_ckpt, name + '_params.pt')

    # load
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary

    # outdir
    os.makedirs(path_gendir, exist_ok=True)

    # config
    n_class = []
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))

    # init model
    net = TransformerModel(n_class, is_training=False)
    net.cuda()
    net.eval()

    # load model
    print('[*] load model from:', path_saved_ckpt)
    # net.load_state_dict(torch.load(path_saved_ckpt, map_location=torch.device('cpu')))
    net.load_state_dict(torch.load(path_saved_ckpt))

    # gen
    start_time = time.time()
    song_time_list = []
    words_len_list = []


    id_to_emotion = ["HAHV", "HALV", "LALV", "LAHV"]
    for emotion_id in range(4):
        print(emotion_id, id_to_emotion[emotion_id])
        sidx = 0
        while sidx < per_class_num_songs:
            try:
                start_time = time.time()
                path_outfile = os.path.join(path_gendir, '{}_{}.mid'.format(id_to_emotion[emotion_id], str(sidx)))

                res = net.inference_from_scratch(dictionary, emotion_kind=emotion_id)
                write_midi(res, path_outfile, word2event)

                song_time = time.time() - start_time
                word_len = len(res)
                print('song time:', song_time)
                print('word_len:', word_len)
                words_len_list.append(word_len)
                song_time_list.append(song_time)

                sidx = sidx + 1

            except KeyboardInterrupt:
                raise ValueError(' [x] terminated.')
            # except:
            #     continue

    print('ave token time:', sum(words_len_list) / sum(song_time_list))
    print('ave song time:', np.mean(song_time_list))

    # runtime_result = {
    #     'song_time': song_time_list,
    #     'words_len_list': words_len_list,
    #     'ave token time:': sum(words_len_list) / sum(song_time_list),
    #     'ave song time': float(np.mean(song_time_list)),
    # }

    # with open(path_gendir + '/runtime_stats.json', 'w') as f:
    #     json.dump(runtime_result, f)


def save_npy():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # path
    path_ckpt = info_load_model[0]  # path to ckpt dir
    loss = info_load_model[1]  # loss
    name = 'loss_' + str(loss)
    path_saved_ckpt = os.path.join(path_ckpt, name + '_params.pt')

    # load
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary

    # outdir
    # os.makedirs(path_gendir, exist_ok=True)

    # config
    n_class = []
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))

    # init model
    net = TransformerModel(n_class, is_training=False)
    net.cuda()
    net.eval()

    # load model
    print('[*] load model from:', path_saved_ckpt)
    # net.load_state_dict(torch.load(path_saved_ckpt, map_location=torch.device('cpu')))
    net.load_state_dict(torch.load(path_saved_ckpt))

    arr_pros = {
        0: None,
        1: None,
        2: None,
        3: None
    }

    for i in range(4):
        print('emotion_kind: ', i)
        emo_arr = []
        for j in range(per_class_num_songs):
            res = net.inference_from_scratch(dictionary, emotion_kind=i)
            res = res.tolist()
            # np.append(emo_arr, res, axis=0)
            emo_arr.append(res)
        emo_arr = np.array(emo_arr)

        print("emo_arr.shape: ", emo_arr.shape)
        arr_pros[i] = emo_arr

    process_kinds_data(arr_pros[0], arr_pros[1], arr_pros[2], arr_pros[3], save_path='./'+path_exp, file_name=path_exp)



if __name__ == '__main__':
    # -- training -- #
    if MODE == 'train':
        train()

    # -- inference -- #
    elif MODE == 'inference':
        generate()

    # -- save npy for classification -- #
    elif MODE == 'save_npy':
        save_npy()

    else:
        pass
