import math
import numpy as np
import torch
import muspy
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import seaborn as sns

BEAT_RESOL = 480


def pitch_count(sequence, min_note=0, max_note=127):
    pitches = [p for p in sequence if p in range(0, max_note - min_note + 1, 2)]
    unique_elements = torch.unique(pitches)
    pc = len(unique_elements)
    return pc

def average_pitch_interval(sequence, min_note=0, max_note=127):
    pitches = [p for p in sequence if p in range(0, max_note - min_note + 1, 2)]
    pitches1 = pitches[:-1]
    pitches2 = pitches[1:]
    pitch_diff = pitches2 - pitches1
    api = np.mean(pitch_diff)

    return api

def average_inter_onset_interval(sequence, min_note=0, max_note=127, num_time_shifts=10):
    shifts = [s for s in sequence if s in range((max_note - min_note + 1), (max_note - min_note + 1) + num_time_shifts)]
    aioi = np.mean(shifts) * (100 // num_time_shifts)

    return aioi

# 计算 note density
def calculate_note_density(midi_file):
    # Load the MIDI file
    midi = muspy.read_midi(midi_file)

    # Calculate the note density per beat
    num_beats = int(midi.get_end_time() / BEAT_RESOL)
    try:
        note_density = sum([len(track.notes) for track in midi.tracks]) / num_beats
        # print('Note density per beat:', note_density)
        return note_density
    except:
        return None

# 计算 note length
def calculate_note_length(midi_file):
    # Load the MIDI file
    midi = muspy.read_midi(midi_file)
    # Iterate over notes and calculate average note length
    total_duration = 0
    total_notes = 0
    for track in midi.tracks:
        for note in track.notes:
            duration = note.duration
            duration_in_beats = duration / (BEAT_RESOL)
            total_duration += duration_in_beats
            total_notes += 1

    average_note_length = total_duration / total_notes
    print("Average note length in beat unit:", average_note_length)
    return average_note_length

def calculate_note_density_arr(midi_file_list):
    note_density_arr = []
    for midi_file in midi_file_list:
        note_density = calculate_note_density(midi_file)
        if note_density is not None: note_density_arr.append(note_density)
    return note_density_arr

def calculate_note_length_arr(midi_file_list):
    note_length_arr = []
    for midi_file in midi_file_list:
        note_length = calculate_note_length(midi_file)
        note_length_arr.append(note_length)
    return note_length_arr


def plot_note_length(data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7):
    data = [data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7]
    labels = ["Bethel", "Clayderman", "Einaudi", "Hancock", "Hillsong", "Hisaishi", "Ryuichi", "Yiruma"]
    fig = plt.figure(figsize=(10, 12))
    sns.violinplot(data=data)
    plt.xticks(range(len(labels)), labels)
    plt.ylabel('Note Length')
    plt.title('Note Length Comparison')
    plt.show()
    fig.savefig('note_length.png')

def plot_note_density(data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7):
    data = [data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7]
    labels = ["Bethel", "Clayderman", "Einaudi", "Hancock", "Hillsong", "Hisaishi", "Ryuichi", "Yiruma"]
    fig = plt.figure(figsize=(10, 12))
    sns.violinplot(data=data)
    plt.xticks(range(len(labels)), labels)
    plt.ylabel('Note Density')
    plt.title('Note Density Comparison')
    plt.show()
    fig.savefig('note_density.png')


def calculate_scores(midi_file, scores_to_calculate=['pitch_range', 'number_pitch_classes', 'polyphony']):
    scores = defaultdict(list)
    midi_obj = muspy.read_midi(midi_file)
    for score in scores_to_calculate:
        if score == 'pitch_range':
            scores['pitch_range'] = muspy.pitch_range(midi_obj)
        elif score == 'number_pitch_classes':
            scores['number_pitch_classes'] = muspy.n_pitch_classes_used(midi_obj)
        elif score == 'polyphony':
            scores['polyphony'] = muspy.polyphony(midi_obj)
        elif score == 'scale_consistency':
            scores['scale_consistency'] = muspy.scale_consistency(midi_obj)
        else:
            print('Score not found.')
    return scores


def calculate_scores_multi(midi_file_list, scores_to_calculate=['pitch_range', 'number_pitch_classes', 'scale_consistency', 'polyphony']):
    scores = []
    for midi_file in midi_file_list:
        scores.append(calculate_scores(midi_file, scores_to_calculate))

    avg_scores = {s_name: np.mean([score[s_name] for score in scores], dtype=np.float32) for s_name in
                  scores_to_calculate}

    return scores, avg_scores


def all_path(dirname):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            result.append(apath)
    return result


if __name__ == '__main__':
    # midi_file_list = all_path('./test_changeData')
    # scores, avg_scores = calculate_scores_multi(midi_file_list)
    # # 原始数据的指标结果：
    # print(avg_scores)  # {'pitch_range': 59.89376, 'number_pitch_classes': 9.783305, 'scale_consistency': 0.9110882}
    #
    # # 分类各自的指标
    # midi_file_list_0 = all_path('./data/0')
    # midi_file_list_1 = all_path('./data/1')
    # midi_file_list_2 = all_path('./data/2')
    # midi_file_list_3 = all_path('./data/3')
    # midi_file_list_4 = all_path('./data/4')
    # midi_file_list_5 = all_path('./data/5')
    # midi_file_list_6 = all_path('./data/6')
    # midi_file_list_7 = all_path('./data/7')
    # scores_0, avg_scores_0 = calculate_scores_multi(midi_file_list_0)
    # scores_1, avg_scores_1 = calculate_scores_multi(midi_file_list_1)
    # scores_2, avg_scores_2 = calculate_scores_multi(midi_file_list_2)
    # scores_3, avg_scores_3 = calculate_scores_multi(midi_file_list_3)
    # scores_4, avg_scores_4 = calculate_scores_multi(midi_file_list_4)
    # scores_5, avg_scores_5 = calculate_scores_multi(midi_file_list_5)
    # scores_6, avg_scores_6 = calculate_scores_multi(midi_file_list_6)
    # scores_7, avg_scores_7 = calculate_scores_multi(midi_file_list_7)
    # print(avg_scores_0)   # {'pitch_range': 57.201492, 'number_pitch_classes': 7.0597014, 'scale_consistency': 0.99971646}
    # print(avg_scores_1)   # {'pitch_range': 69.338234, 'number_pitch_classes': 9.926471, 'scale_consistency': 0.94114596}
    # print(avg_scores_2)   # {'pitch_range': 46.91129, 'number_pitch_classes': 8.16129, 'scale_consistency': 0.97926676}
    # print(avg_scores_3)   # {'pitch_range': 66.88571, 'number_pitch_classes': 11.946428, 'scale_consistency': 0.78120786}
    # print(avg_scores_4)   # {'pitch_range': 58.126762, 'number_pitch_classes': 7.352113, 'scale_consistency': 0.9977595}
    # print(avg_scores_5)   # {'pitch_range': 59.069767, 'number_pitch_classes': 10.658915, 'scale_consistency': 0.92115706}
    # print(avg_scores_6)   # {'pitch_range': 56.293335, 'number_pitch_classes': 10.633333, 'scale_consistency': 0.89522123}
    # print(avg_scores_7)   # {'pitch_range': 55.78022, 'number_pitch_classes': 10.285714, 'scale_consistency': 0.91902405}


    # note density 与 note length 的 violin 图

    midi_file_list_0 = all_path('./data/0')
    midi_file_list_1 = all_path('./data/1')
    midi_file_list_2 = all_path('./data/2')
    midi_file_list_3 = all_path('./data/3')
    midi_file_list_4 = all_path('./data/4')
    midi_file_list_5 = all_path('./data/5')
    midi_file_list_6 = all_path('./data/6')
    midi_file_list_7 = all_path('./data/7')

    note_density_0, note_density_1, note_density_2, note_density_3, note_density_4, note_density_5, note_density_6, note_density_7 \
        = calculate_note_density_arr(
        midi_file_list_0), calculate_note_density_arr(midi_file_list_1), calculate_note_density_arr(
        midi_file_list_2), calculate_note_density_arr(midi_file_list_3), calculate_note_density_arr(
        midi_file_list_4), calculate_note_density_arr(midi_file_list_5), calculate_note_density_arr(
        midi_file_list_6), calculate_note_density_arr(midi_file_list_7)

    plot_note_density(note_density_0, note_density_1, note_density_2, note_density_3, note_density_4, note_density_5, note_density_6, note_density_7)

    note_length_0, note_length_1, note_length_2, note_length_3, note_length_4, note_length_5, note_length_6, note_length_7 \
        = calculate_note_length_arr(
        midi_file_list_0), calculate_note_length_arr(midi_file_list_1), calculate_note_length_arr(
        midi_file_list_2), calculate_note_length_arr(midi_file_list_3), calculate_note_length_arr(
        midi_file_list_4), calculate_note_length_arr(midi_file_list_5), calculate_note_length_arr(
        midi_file_list_6), calculate_note_length_arr(midi_file_list_7)

    plot_note_length(note_length_0, note_length_1, note_length_2, note_length_3, note_length_4, note_length_5, note_length_6, note_length_7)
    """
    The polyphony is defined as the average number of pitches being played at the same time, 
    evaluated only at time steps where at least one pitch is on. Drum tracks are ignored. Return NaN if no note is found
    """

    """
    The scale consistency is defined as the largest pitch-in-scale rate over all major and minor scales
    """