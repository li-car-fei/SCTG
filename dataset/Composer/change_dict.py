"""
改变字典Dict
原先的： ['Bar', 'Position', 'Pitch', 'Duration']
需要新增2个 'Type' 'Composer' ['Bar', 'Position', 'Pitch', 'Duration', 'Type', 'Composer']
"""

"""
'Type' : { "CP" : 0, "Composer" : 1 }
'Composer' : { "Bethel" : 0, "Clayderman" : 1, "Einaudi" : 2, "Hancock" : 3, "Hillsong": 4 ,
             "Hisaishi": 5, "Ryuichi": 6, "Yiruma": 7, "<PAD>": 8 }

当 'Type'=0   ['Bar', 'Position', 'Pitch', 'Duration', 0, 8]
当 'Type'=1   ['Bar <PAD>', 'Position <PAD>', 'Pitch <PAD>', 'Duration <PAD>', 1, 0||1||2||3||4||5||6||7]

在原先数据集的头部添加对应的 ['Bar <PAD>', 'Position <PAD>', 'Pitch <PAD>', 'Duration <PAD>', 1, 0||1||2||3||4||5||6||7]  
                        即 [2, 16, 86, 64, 1, 0||1||2||3||4||5||6||7 ]
最后的情感标签需要与answer中对应

原先的 1024 序列最后一个不截掉，然后1-1024的[]都要在末端加上[0, 8]
"""

Composer = {
    "Bethel": 0,
    "Clayderman": 1,
    "Einaudi": 2,
    "Hancock": 3,
    "Hillsong": 4,
    "Hisaishi": 5,
    "Ryuichi": 6,
    "Yiruma": 7,
    "<Pad>": 8,
}

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


def look_data(root='./composer-CP', dataset='composer_cp_train_new'):
    data = np.load(os.path.join(root, f'{dataset}.npy'), allow_pickle=True)
    print(f'   {dataset}: {data.shape}')                        # emopia_cp_train: (924, 512, 4)   只用了4个token ['Bar', 'Position', 'Pitch', 'Duration']
    print(data[0])
    for i in data[619]:
        print(i)


# 转变dict
def dict():
    ori_e2w, ori_w2e = get_dict('./composer-CP/CP.pkl')

    ori_e2w['Type'] = {"CP": 0, "Composer": 1 }
    ori_e2w['Composer'] = {"Bethel": 0, "Clayderman": 1, "Einaudi": 2, "Hancock": 3, "Hillsong": 4,
                            "Hisaishi": 5, "Ryuichi": 6, "Yiruma": 7, "<PAD>": 8}

    ori_w2e['Type'] = {0: "CP", 1: "Composer"}
    ori_w2e['Composer'] = {0: "Bethel", 1: "Clayderman", 2: "Einaudi", 3: "Hancock", 4: "Hillsong",
                          5: "Hisaishi", 6: "Ryuichi", 7: "Yiruma", 8: "<PAD>"}

    path_new_dict = './composer-CP/CP_new.pkl'
    pickle.dump((ori_e2w, ori_w2e), open(path_new_dict, 'wb'))

    get_dict(path_new_dict)


def get_data(root='./composer-CP', dataset='composer_train'):
    data = np.load(os.path.join(root, f'{dataset}.npy'), allow_pickle=True)
    print(f'   {dataset}: {data.shape}')
    print(data)
    return data

def get_answer_data(root='./composer-CP', dataset='composer_train_ans'):
    data = np.load(os.path.join(root, f'{dataset}.npy'), allow_pickle=True)
    # print(f'   {dataset}: {data.shape}')
    # print(data)
    return data

def change_data():
    ori_data = get_data(root='./composer-CP', dataset='composer_train')                 # composer_train: composer_train: (1186, 512, 4)
    answer_data = get_answer_data(root='./composer-CP', dataset='composer_train_ans')

    sample_len = ori_data.shape[0]
    seq_len = ori_data.shape[1]

    answer_list = answer_data.tolist()
    ori_list = ori_data.tolist()
    new_data = []

    for i in range(sample_len):
        new_sample = ori_list[i]
        # new_sample.pop()                    #  去除最后一个

        for j in range(seq_len):              # 新增 type 与 composer token
            new_sample[j].append(0)
            new_sample[j].append(answer_list[i])           # composer标签   当只有头部标签时，则是插入8即<Pad>

        new_sample.insert(0, [2, 16, 86, 64, 1, answer_list[i]])    # 头部插入

        new_data.append(new_sample)

    new_data = np.array(new_data)
    save_new_data_path = './composer-CP/composer_cp_train_new.npy'
    np.save(save_new_data_path, new_data)

    get_data(root='./composer-CP', dataset='composer_cp_train_new')         # composer_cp_train_new: (1186, 513, 6)


# 制作泛化能力数据集，从 composer_cp_train_new.npy 中分割出来即可
# generalization/composer_cp_train_without_(composer_name)    训练阶段一，没有某个作曲家的数据
# generalization/composer_cp_finetune_with_(composer_name)    训练阶段二，具体泛化某个作曲家的数据
def create_generalization_data(composer_name):
    if not os.path.exists('./generalization'):
        os.mkdir('./generalization')
        print("Folder created:", './generalization')
    else:
        print("Folder already exists:", './generalization')

    all_data = get_data(root='./composer-CP',
                        dataset='composer_cp_train_new')  # composer_train: composer_train: (1186, 512, 4)
    answer_data = get_answer_data(root='./composer-CP', dataset='composer_train_ans')

    # 某个作曲家对应的label
    composer_label = Composer[composer_name]

    sample_len = all_data.shape[0]
    answer_list = answer_data.tolist()
    all_list = all_data.tolist()
    train_data = []                                             # 训练阶段一 数据
    train_data_answer = []                                      # 训练阶段一 answer 标签
    finetune_data = []                                          # 训练阶段二 数据
    finetune_data_answer = []                                   # 训练阶段二 answer 标签

    for i in range(sample_len):
        cur_composer_label = answer_list[i]
        if cur_composer_label == composer_label:
            print('current composer label: ', cur_composer_label)
            finetune_data.append(all_list[i])
            finetune_data_answer.append(cur_composer_label)
        else:
            train_data.append(all_list[i])
            train_data_answer.append(cur_composer_label)

    train_data = np.array(train_data)
    save_train_data_path = './generalization/composer_cp_train_without_' + composer_name + '.npy'
    np.save(save_train_data_path, train_data)

    train_data_answer = np.array(train_data_answer)
    save_train_data_answer_path = './generalization/answer_train_without_' + composer_name + '.npy'
    np.save(save_train_data_answer_path, train_data_answer)

    finetune_data = np.array(finetune_data)
    save_finetune_data_path = './generalization/composer_cp_finetune_with_' + composer_name + '.npy'
    np.save(save_finetune_data_path, finetune_data)

    finetune_data_answer = np.array(finetune_data_answer)
    save_finetune_data_answer_path = './generalization/answer_finetune_with_' + composer_name + '.npy'
    np.save(save_finetune_data_answer_path, finetune_data_answer)



if __name__ == '__main__':
    get_data('./composer-CP', 'composer_cp_train_new')          # composer_train: (1186, 513, 6)

    get_dict('./composer-CP/CP_new.pkl')
    # dict()
    # change_data()

    # 探索泛化能力 generalization ability
    # 用七个作曲家的歌曲先正常训练模型，再用剩下一个作曲家的数据微调模型
    # 让微调后的模型生成后面那个作曲家的音乐
    # 结果应当和正常八个作曲家音乐训练的模型输出进行对比，如果结果评估差不多，那么说明泛化能力可以
    create_generalization_data(composer_name="Yiruma")
    create_generalization_data(composer_name="Ryuichi")

    #


