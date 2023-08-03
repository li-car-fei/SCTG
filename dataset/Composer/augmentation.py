# 数据增强
# 直接对 composer_cp_train_new.npy 中的数据做 pitch shift

import pickle
import random
from copy import deepcopy
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

def get_data(root='./composer-CP', dataset='composer_train'):
    data = np.load(os.path.join(root, f'{dataset}.npy'), allow_pickle=True)
    print(f'   {dataset}: {data.shape}')
    # print(data)
    return data

def get_answer_data(root='./composer-CP', dataset='composer_train_ans'):
    data = np.load(os.path.join(root, f'{dataset}.npy'), allow_pickle=True)
    # print(f'   {dataset}: {data.shape}')
    # print(data)
    return data


"""
    pitch [22, 107]   对应字典 [0, 85]
    流程：
        1 数据中提取出除了头部标签之外的pitch，长度512
        2 求这个序列的 min 和 max
        3 因为数据中已经是字典中的符号 即 [0, 85]
            当：最小值减去6，如果大于等于0，则可以随机整体降调 [-6, -1] 的值
               且 最大值加5，如果小于等于85，则可以整体升调 [+1, +5]
            则：随机 偏移调 [-6, +5]
        4 新得到的pitch序列和原先其他合并为原来的结构，注意头部标签插回头部，作为新数据
        5 获得新数据时注意要取出它们对应的label标签，后面训练要用
        6 新数据和老数据合并，作为新的训练数据集

"""
def augmentation():
    if not os.path.exists('./augmentation'):
        os.mkdir('./augmentation')
        print("Folder created:", './augmentation')
    else:
        print("Folder already exists:", './augmentation')

    ori_all_data = get_data(root='./composer-CP',
                        dataset='composer_cp_train_new')  # composer_train: composer_train: (1186, 513, 4)
    ori_answer_data = get_answer_data(root='./composer-CP', dataset='composer_train_ans')

    ori_sample_len = ori_all_data.shape[0]
    print('original sample length: ', ori_sample_len, ori_answer_data.shape[0])
    # original sample length:  1186 1186

    ori_all_list = ori_all_data.tolist()                                            # 原数据列表
    ori_answer_list = ori_answer_data.tolist()                                      # 原数据对应的answer

    aug_data_list = []                                                              # 增强得到的数据
    aug_answer_list = []                                                            # 增强数据对应的answer

    for i in range(ori_sample_len):
        cur_sample = ori_all_list[i]                                                # 当前样本数据
        cur_answer = ori_answer_list[i]                                             # 当前样本对应的answer标签
        cur_sample_cp = cur_sample[1:]                                              # 当前样本，不包括头部标签

        cur_pitch_list = [cur_cp[2] for cur_cp in cur_sample_cp]                    # 当前样本 pitch 列
        cur_max_pitch, cur_min_pitch = max(cur_pitch_list), min(cur_pitch_list)     # pitch 最大最小值

        # if cur_min_pitch - 6 >= 0:
        #     random_shift_pitch_value = random.randint(1, 6)
        #     print("pitch shift: ", -random_shift_pitch_value)
        #     shift_pitch_list = [pitch - random_shift_pitch_value for pitch in cur_pitch_list]
        #     aug_sample = deepcopy(cur_sample)
        #     for j in range(1, len(aug_sample)):                                     # 更改pitch，不改变头部的标签
        #         aug_sample[j][2] = shift_pitch_list[j-1]                            # shift_pitch_list 没有头部标签，是不对齐的
        #
        #     aug_data_list.append(aug_sample)
        #     aug_answer_list.append(cur_answer)
        #
        # if cur_max_pitch + 5 <= 85:
        #     random_shift_pitch_value = random.randint(1, 5)
        #     print("pitch shift: ", random_shift_pitch_value)
        #     shift_pitch_list = [pitch + random_shift_pitch_value for pitch in cur_pitch_list]
        #     aug_sample = deepcopy(cur_sample)
        #     for j in range(1, len(aug_sample)):                                     # 更改pitch，不改变头部的标签
        #         aug_sample[j][2] = shift_pitch_list[j-1]                            # shift_pitch_list 没有头部标签，是不对齐的
        #
        #     aug_data_list.append(aug_sample)
        #     aug_answer_list.append(cur_answer)

        if cur_min_pitch - 6 >= 0 and cur_max_pitch + 5 <= 85:
            random_shift_pitch_value = random.randint(-6, 5)
            print("pitch shift: ", random_shift_pitch_value)
            shift_pitch_list = [pitch + random_shift_pitch_value for pitch in cur_pitch_list]
            aug_sample = deepcopy(cur_sample)
            for j in range(1, len(aug_sample)):                                     # 更改pitch，不改变头部的标签
                aug_sample[j][2] = shift_pitch_list[j-1]                            # shift_pitch_list 没有头部标签，是不对齐的

            aug_data_list.append(aug_sample)
            aug_answer_list.append(cur_answer)

    save_data = ori_all_list + aug_data_list
    save_answer = ori_answer_list + aug_answer_list

    print('after augmentation sample length: ', len(save_data), len(save_answer))
    # after augmentation sample length:  3040 3040

    save_data = np.array(save_data)
    save_data_path = './augmentation/composer_cp_train' + '.npy'
    np.save(save_data_path, save_data)

    save_answer = np.array(save_answer)
    save_answer_path = './augmentation/answer_train' + '.npy'
    np.save(save_answer_path, save_answer)






if __name__ == '__main__':
    # get_data('./composer-CP', 'composer_cp_train_new')          # composer_train: (1186, 513, 6)
    # get_dict('./composer-CP/CP_new.pkl')

    # 数据增强 pitch shift
    augmentation()



