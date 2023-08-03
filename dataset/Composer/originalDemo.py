import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note
import pickle
import numpy as np
import os

BEAT_RESOL = 480
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

def look_dict(dict_file):
    with open(dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)
        print(e2w)
        print("#" * 20)
        print(w2e)
        return e2w, w2e

def write_some_midis(word2event, root='./composer-CP', dataset='composer_cp_train_new', answer='composer_train_ans'):
    data = np.load(os.path.join(root, f'{dataset}.npy'), allow_pickle=True)
    data_answer = np.load(os.path.join(root, f'{answer}.npy'))
    print(f'   {dataset}: {data.shape}')
    print(f'   {answer}: {data_answer.shape}')

    for data_id in range(1186):
        cur_data = data[data_id]
        path_outfile = './test_changeData/' + str(data_id) + '.mid'
        write_midi(cur_data, path_outfile, word2event)

# 根据标签得到midis，分别统计
def label_midis(word2event, root='./composer-CP', dataset='composer_cp_train_new', answer='composer_train_ans'):
    data = np.load(os.path.join(root, f'{dataset}.npy'), allow_pickle=True)
    data_answer = np.load(os.path.join(root, f'{answer}.npy'))
    print(f'   {dataset}: {data.shape}')
    print(f'   {answer}: {data_answer.shape}')

    emotion_to_path = {
        0: './data/0/',
        1: './data/1/',
        2: './data/2/',
        3: './data/3/',
        4: './data/4/',
        5: './data/5/',
        6: './data/6/',
        7: './data/7/',
    }

    for idx in range(1186):
        cur_emotion = data_answer[idx]
        cur_data = data[idx]
        path_outfile = emotion_to_path[cur_emotion] + str(idx) + '.mid'
        write_midi(cur_data, path_outfile, word2event)



if __name__ == '__main__':
    e2w, w2e = look_dict('./composer-CP/CP_new.pkl')
    write_some_midis(w2e)
    label_midis(w2e)