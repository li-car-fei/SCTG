import pickle
import numpy as np
import os


def process_data_npy_no_split(arr_data, cls_ans):
    """
    arr_data     np     [sample_num, 513, 6]
    ans_ans      style
    """

    sample_num = arr_data.shape[0]
    seq_len = arr_data.shape[1]                             

    ori_data = arr_data.tolist()
    new_data = []

    new_cls_ans = []

    for i in range(sample_num):
        sample = ori_data[i]
        # sample.pop(0)                                  

        new_sample = []
        for j in range(seq_len):
            new_note = []
            new_note.append(sample[j][0])                 # bar
            new_note.append(sample[j][1])                 # position
            new_note.append(sample[j][2])                 # pitch
            new_note.append(sample[j][3])                 # duration
            new_sample.append(new_note)

        new_data.append(new_sample)
        new_cls_ans.append(cls_ans)

    new_data = np.array(new_data)
    new_cls_ans = np.array(new_cls_ans)

    return new_data, new_cls_ans




def process_data_npy(arr_data, cls_ans):
    """
    arr_data     np     [sample_num, 513, 6]
    ans_ans      style
    """

    sample_num = arr_data.shape[0]
    seq_len = arr_data.shape[1] - 1                         

    ori_data = arr_data.tolist()
    new_data = []

    new_cls_ans = []

    for i in range(sample_num):
        sample = ori_data[i]
        sample.pop(0)                                       

        new_sample = []
        for j in range(seq_len):
            new_note = []
            new_note.append(sample[j][0])                 # bar
            new_note.append(sample[j][1])                 # position
            new_note.append(sample[j][2])                 # pitch
            new_note.append(sample[j][3])                 # duration
            new_sample.append(new_note)

        new_data.append(new_sample)
        new_cls_ans.append(cls_ans)

    new_data = np.array(new_data)
    new_cls_ans = np.array(new_cls_ans)

    return new_data, new_cls_ans


def process_kinds_data(arr_data_0, arr_data_1, arr_data_2, arr_data_3, arr_data_4, arr_data_5, arr_data_6, arr_data_7, save_path, file_name):
    new_data_0, new_ans_0 = process_data_npy(arr_data_0, 0)
    new_data_1, new_ans_1 = process_data_npy(arr_data_1, 1)
    new_data_2, new_ans_2 = process_data_npy(arr_data_2, 2)
    new_data_3, new_ans_3 = process_data_npy(arr_data_3, 3)
    new_data_4, new_ans_4 = process_data_npy(arr_data_4, 4)
    new_data_5, new_ans_5 = process_data_npy(arr_data_5, 5)
    new_data_6, new_ans_6 = process_data_npy(arr_data_6, 6)
    new_data_7, new_ans_7 = process_data_npy(arr_data_7, 7)

    new_data = np.concatenate((new_data_0, new_data_1, new_data_2, new_data_3, new_data_4, new_data_5, new_data_6, new_data_7), axis=0)
    new_ans = np.concatenate((new_ans_0, new_ans_1, new_ans_2, new_ans_3, new_ans_4, new_ans_5, new_ans_6, new_ans_7), axis=0)

    print("new_data.shape: ", new_data.shape)
    print("new_ans.shape", new_ans.shape)

    save_new_data_path = save_path + '/' + file_name +'_testData.npy'
    save_new_ans_path = save_path + '/' + file_name + '_testData_ans.npy'

    np.save(save_new_data_path, new_data)
    np.save(save_new_ans_path, new_ans)



def look_data(root='./only_nll', dataset='testData'):
    data = np.load(os.path.join(root, f'{dataset}.npy'), allow_pickle=True)
    print(f'   {dataset}: {data.shape}')                        # emopia_cp_train: (924, 512, 4)   只用了4个token ['Bar', 'Position', 'Pitch', 'Duration']
    print(data)


def look_answer_data(root='./only_nll', dataset='testData_ans'):
    data = np.load(os.path.join(root, f'{dataset}.npy'), allow_pickle=True)
    print(f'   {dataset}: {data.shape}')                    #  emopia_cp_train_ans: (924,)
    print(data)



if __name__ == '__main__':
    look_answer_data()
    look_data()


