import pickle
import numpy as np
import os

def look_dict(dict_file):
    with open(dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

        print(e2w)
        print("#" * 20)
        print(w2e)

def look_data(root='./emopia-CP', dataset='emopia_cp_train'):
    data = np.load(os.path.join(root, f'{dataset}.npy'), allow_pickle=True)
    print(f'   {dataset}: {data.shape}')                        # emopia_cp_train: (924, 512, 4)   只用了4个token ['Bar', 'Position', 'Pitch', 'Duration']
    print(data)


def look_answer_data(root='./emopia-CP', dataset='emopia_cp_train_ans'):
    data = np.load(os.path.join(root, f'{dataset}.npy'), allow_pickle=True)
    print(f'   {dataset}: {data.shape}')                    #  emopia_cp_train_ans: (924,)
    print(data)


def look_train_data(root='./emopia-CP', dataset='train_data_linear'):
    data = np.load(os.path.join(root, f'{dataset}.npz'), allow_pickle=True)
    train_x = data['x']
    train_y = data['y']
    train_mask = data['mask']
    print(train_x.shape)                    # (924, 512, 4)
    print(train_y.shape)                    # (924, 512, 4)
    print(train_mask.shape)                 # (924, 512)
    print(train_x)
    print('#'*40)
    print(train_y)

if __name__ == '__main__':
    look_dict('./emopia-CP/CP_dict.pkl')
    # look_data()
    # look_answer_data()
    # look_train_data()