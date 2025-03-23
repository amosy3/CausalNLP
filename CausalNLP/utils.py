import pickle
import datetime
import os
import torch


class Logger:
    def __init__(self, path='./logs/', filename=''):
        t0 = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = ('%s/%s_%s/' % (path, filename, t0)).replace(' ', '_')
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_file = self.log_dir + 'log'

        os.mkdir(self.log_dir + 'checkpoints/')
        self.log_models = self.log_dir + 'checkpoints/'

        print('Log directory created at %s' % self.log_dir)

    def print_and_log(self, txt, just_log=False):
        t0 = datetime.datetime.now()
        txt = '[%s]: %s' % (t0, txt)
        if not(just_log):
            print(txt)
        f = open(self.log_file, 'a')
        f.write(txt + '\n')
        f.close()

    def log_model(self, model, add_to_name=''):
        # t1 = datetime.datetime.now()
        torch.save(model.state_dict(), '%s_%s.pt' % (self.log_models, add_to_name))

    def log_object(self, obj, obj_name=''):
        save_object(obj, '%s%s.pkl' % (self.log_dir, obj_name))


def title_print(txt, wide=20):
    print('\n' + '#' * wide)
    pad = (wide - (len(txt)+2)) // 2
    fix = (wide - (len(txt)+2)) - 2 * pad
    print('#' + ' ' * (pad+fix) + txt + ' ' * pad + '#')
    print('#' * wide + '\n')


def save_object(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_object(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj