# 根据MIDI-BERT的处理好的数据完成CP训练所需的数据结构 train_x train_y train_mask

import os
import json
import pickle
import numpy as np


TEST_AMOUNT = 50
WINDOW_SIZE = 512
GROUP_SIZE = 6                        # ['Bar', 'Position', 'Pitch', 'Duration', 'Type', 'Emo']

"""
MIDI-BERT 的数据处理好的都是512+1长度的，因此MAX_LEN不需要设置为怎么大了，后面都是补0而已
"""
MAX_LEN = WINDOW_SIZE + 1


COMPILE_TARGET = 'linear' # 'linear', 'XL'
print('[config] MAX_LEN:', MAX_LEN)

if __name__ == '__main__':
    # paths
    path_root = 'emopia-CP'
    data_indir = os.path.join(path_root, 'emopia_cp_train_new.npy')

    # load dictionary
    path_dictionary = os.path.join(path_root, 'CP_dict_new.pkl')
    event2word, word2event = pickle.load(open(path_dictionary, 'rb'))

    # load all words
    dataset = np.load(data_indir)
    print(f': {dataset.shape}')

    # load answer data
    def get_answer_data(root='./emopia-CP', dataset='emopia_cp_train_ans'):
        data = np.load(os.path.join(root, f'{dataset}.npy'), allow_pickle=True)
        # print(f'   {dataset}: {data.shape}')                    #    composer_cp_train_ans: (1186,)
        # print(data)
        return data
    answer_data = get_answer_data(root='./emopia-CP', dataset='emopia_cp_train_ans')

    # init
    x_list = []
    y_list = []
    mask_list = []
    seq_len_list = []
    num_groups_list = []
    name_list = []

    # process
    for fidx in range(dataset.shape[0]):
        words = dataset[fidx]
        num_words = words.shape[0]
        # eos_arr = words[-1][None, ...]                        # 前一步骤在末尾添加的 eso event                            # 前一步骤在末尾添加的 eso event
        eos_arr = [[2, 16, 86, 64, 0, answer_data[fidx]]]                       # 最后一个是Emo，当只有头部标签时则是4即<Pad>
        print(' words_shape:', words.shape)     # words_shape: (513, 4)

        # 输入是固定长度的512，不需要判定了
        # if num_words >= MAX_LEN - 2:  # 2 for room
        #     print(' [!] too long:', num_words)
        #     continue

        # arrange IO
        x = words[:-1].copy()
        y = words[1:].copy()
        seq_len = len(x)
        print(' > x_shape:', x.shape)           # x_shape: (512, 4)
        print(' > y_shape:', y.shape)           # y_shape: (512, 4)
        print(' > seq_len:', seq_len)           # seq_len: 512

        # pad with eos
        pad = np.tile(
            eos_arr,
            (MAX_LEN - seq_len, 1))
        print(' > pad_shape:', pad.shape)       # pad_shape: (1, 6)


        x = np.concatenate([x, pad], axis=0)
        y = np.concatenate([y, pad], axis=0)
        mask = np.concatenate(
            [np.ones(seq_len), np.zeros(MAX_LEN - seq_len)])
        print(' > x_shape:', x.shape)                           # x_shape: (513, 6)
        print(' > y_shape:', y.shape)                           # y_shape: (513, 6)
        print(' > mask_shape:', mask.shape)                     # mask_shape: (513, )

        # collect
        x_list.append(x)
        y_list.append(y)
        mask_list.append(mask)
        seq_len_list.append(seq_len)
        num_groups_list.append(int(np.ceil(seq_len / WINDOW_SIZE)))                     # 序列长度与window相除结果
        name_list.append(fidx)

    # sort by length (descending)
    zipped = zip(seq_len_list, x_list, y_list, mask_list, num_groups_list, name_list)
    seq_len_list, x_list, y_list, mask_list, num_groups_list, name_list = zip(
        *sorted(zipped, key=lambda x: -x[0]))

    print('\n\n[Finished]')
    print(' compile target:', COMPILE_TARGET)
    if COMPILE_TARGET == 'XL':
        x_final = np.array(x_list).reshape(-1, GROUP_SIZE, WINDOW_SIZE)
        y_final = np.array(y_list).reshape(-1, GROUP_SIZE, WINDOW_SIZE)
        mask_final = np.array(mask_list).reshape(-1, GROUP_SIZE, WINDOW_SIZE)
    elif COMPILE_TARGET == 'linear':
        x_final = np.array(x_list)
        y_final = np.array(y_list)
        mask_final = np.array(mask_list)
    else:
        raise ValueError('Unknown target:', COMPILE_TARGET)

    # check
    num_samples = len(seq_len_list)
    print(' >   count:', )
    print(' > x_final:', x_final.shape)                             # x_final: (924, 513, 6)
    print(' > y_final:', y_final.shape)                             # y_final: (924, 513, 6)
    print(' > mask_final:', mask_final.shape)                       # mask_final: (924, 513)

    # split train/test
    # validation_songs = json.load(open('../myMusic_validation_songs.json', 'r'))
    train_idx = []
    test_idx = []

    # validation filename map
    fn2idx_map = {
        'fn2idx': dict(),
        'idx2fn': dict(),
    }

    # run split
    valid_cnt = 0
    for nidx, n in enumerate(name_list):
        test_idx.append(nidx)
        train_idx.append(nidx)
    test_idx = np.array(test_idx)
    train_idx = np.array(train_idx)


    # save train
    path_train = os.path.join(path_root, 'train_data_{}'.format(COMPILE_TARGET))
    path_train += '.npz'
    np.savez(
        path_train,
        x=x_final[train_idx],
        y=y_final[train_idx],
        mask=mask_final[train_idx],
        seq_len=np.array(seq_len_list)[train_idx],
        num_groups=np.array(num_groups_list)[train_idx]
    )

    # save test
    path_test = os.path.join(path_root, 'test_data_{}'.format(COMPILE_TARGET))
    path_test += '.npz'
    np.savez(
        path_test,
        x=x_final[test_idx],
        y=y_final[test_idx],
        mask=mask_final[test_idx],
        seq_len=np.array(seq_len_list)[test_idx],
        num_groups=np.array(num_groups_list)[test_idx]
    )

    print('---')
    print(' > train x:', x_final[train_idx].shape)              # train x: (924, 513, 6)
    print(' >  test x:', x_final[test_idx].shape)



