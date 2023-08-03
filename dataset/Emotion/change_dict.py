"""
改变字典Dict
原先的： ['Bar', 'Position', 'Pitch', 'Duration']
需要新增2个 'Type' 'Emotion' ['Bar', 'Position', 'Pitch', 'Duration', 'Type', 'Emotion']
"""

"""
'Type' : { "CP" : 0, "Emo" : 1 }
'Emotion' : { "HAHV" : 0, "HALV" : 1, "LALV" : 2, "LAHV" : 3, "<PAD>": 4 }

当 'Type'=0   ['Bar', 'Position', 'Pitch', 'Duration', 0, 4]
当 'Type'=1   ['Bar <PAD>', 'Position <PAD>', 'Pitch <PAD>', 'Duration <PAD>', 1, 0||1||2||3]

在原先数据集的头部添加对应的 ['Bar <PAD>', 'Position <PAD>', 'Pitch <PAD>', 'Duration <PAD>', 1, 0||1||2||3]  即 [2, 16, 86, 64, 1, 0||1||2||3 ]
最后的情感标签需要与answer中对应

原先的 512序列最后一个截掉，然后1-511的[]都要在末端加上[0, 4]
"""

import pickle
import numpy as np
import os

def get_dict(dict_file):
    with open(dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)
        print("events2words: ", e2w)
        print("#" * 20)
        print("words2events: ", w2e)
        print("#" * 20)

        return e2w, w2e


def dict():
    ori_e2w, ori_w2e = get_dict('./emopia-CP/CP_dict.pkl')

    ori_e2w['Type'] = {"CP": 0, "Emo": 1}
    ori_e2w['Emotion'] = {"HAHV": 0, "HALV": 1, "LALV": 2, "LAHV": 3, "<PAD>": 4}

    ori_w2e['Type'] = {0: "CP", 1: "Emo"}
    ori_w2e['Emotion'] = {0: "HAHV", 1: "HALV", 2: "LALV", 3: "LAHV", 4: "<PAD>"}

    path_new_dict = './emopia-CP/CP_dict_new.pkl'
    pickle.dump((ori_e2w, ori_w2e), open(path_new_dict, 'wb'))

    get_dict(path_new_dict)


def get_data(root='./emopia-CP', dataset='emopia_cp_train'):
    data = np.load(os.path.join(root, f'{dataset}.npy'), allow_pickle=True)
    print(f'   {dataset}: {data.shape}')                                          # emopia_cp_train_new: (924, 513, 6)
    return data

def get_answer_data(root='./emopia-CP', dataset='emopia_cp_train_ans'):
    data = np.load(os.path.join(root, f'{dataset}.npy'), allow_pickle=True)
    # print(f'   {dataset}: {data.shape}')                    #  emopia_cp_train_ans: (924,)
    # print(data)
    return data

def change_data():
    ori_data = get_data(root='./emopia-CP', dataset='emopia_cp_train')
    answer_data = get_answer_data(root='./emopia-CP', dataset='emopia_cp_train_ans')

    sample_len = ori_data.shape[0]
    seq_len = ori_data.shape[1]

    answer_list = answer_data.tolist()
    ori_list = ori_data.tolist()
    new_data = []

    for i in range(sample_len):
        new_sample = ori_list[i]
        # new_sample.pop()                    #  去除最后一个

        for j in range(seq_len):              # 新增 type 与 emotion token
            new_sample[j].append(0)
            new_sample[j].append(answer_list[i])           # Emo标签   当只有头部标签时，则是插入4即<Pad>

        new_sample.insert(0, [2, 16, 86, 64, 1, answer_list[i]])

        new_data.append(new_sample)

    new_data = np.array(new_data)
    save_new_data_path = './emopia-CP/emopia_cp_train_new.npy'
    np.save(save_new_data_path, new_data)

    get_data(root='./emopia-CP', dataset='emopia_cp_train_new')


if __name__ == '__main__':
    get_data()
    get_dict('./emopia-CP/CP_dict_new.pkl')

    # dict()
    # change_data()
