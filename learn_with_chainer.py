# First, we initialise an array for containing state names and a matrix
# for storing voice data and correct labels. Next, read sound files matching
# the pattern of the path for 30 seconds in order, normalize, obtain STFT,
# and register together with the same number of correct answer labels.

import numpy as np
import glob
import librosa

SAMPLING_RATE = 16000
FFT_SIZE = 256
STFT_MATRIX_SIZE = 1 + FFT_SIZE // 2

state_names = []
data = np.empty((0, STFT_MATRIX_SIZE), dtype=np.float32)
index = np.empty(0, dtype=np.int32)

for path_name in sorted(glob.glob('data/*/*.wav')):
    state_name = path_name.split('/')[1]

    if state_name not in state_names:
        state_names.append(state_name)

    audio, sr = librosa.load(path_name, sr=SAMPLING_RATE, duration=30)
    print('{}: {} ({} Hz) '.format(state_name, path_name, sr))
    d = np.abs(librosa.stft(librosa.util.normalize(audio),
                            n_fft=FFT_SIZE, window='hamming'))
    data = np.vstack((data, d.transpose()))
    index = np.hstack([index, [state_names.index(state_name)] * d.shape[1]])

n_states = len(state_names)


# Now that data is ready, we define a neural network with Chainer,
# generate a model, and prepare a dataset with data and correct index as a set.
# Of the data set, we will use 80% for learning and the remaining 20%
# for verification.

import chainer
import chainer.links as L
import chainer.functions as F


class LaserMachineListener(chainer.Chain):

    def __init__(self, n_mid_units, n_out):
        super(LaserMachineListener, self).__init__()

        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(None, n_mid_units // 2)
            self.l3 = L.Linear(None, n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


net = LaserMachineListener(n_mid_units=128, n_out=n_states)
model = L.Classifier(net)

dataset = chainer.datasets.TupleDataset(data, index)
n_train = int(len(dataset) * 0.8)
train, test = chainer.datasets.split_dataset_random(dataset, n_train, seed=0)


# Select Adam as the optimisation method set the number of epochs to 10,
# configure to report progress in progress during learning, and execute learning.
# When using files bundled as samples, the correct answer rate is about 98%.
# If you do not get the proper answer rate for the samples prepared by yourself,
# try tuning parameters such as `n_mid_units` and `BATCH_SIZE`.

from chainer import optimizers
from chainer import training

optimizer = optimizers.Adam(alpha=0.01)
optimizer.setup(model)

GPU_ID = -1

# Specify the seed value of the random number
# so that the same result will be obtained each time
np.random.seed(1)

BATCH_SIZE = 32

train_iter = chainer.iterators.SerialIterator(train, BATCH_SIZE)
test_iter = chainer.iterators.SerialIterator(test, BATCH_SIZE,
                                             repeat=False, shuffle=False)
updater = training.StandardUpdater(train_iter, optimizer, device=GPU_ID)

MAX_EPOCH = 20

trainer = training.Trainer(updater, (MAX_EPOCH, 'epoch'), out='result')

from chainer.training import extensions

trainer.extend(extensions.LogReport())
trainer.extend(extensions.Evaluator(
    test_iter, model, device=GPU_ID), name='val')
trainer.extend(extensions.PrintReport(
    ['epoch',
     'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy',
     'elapsed_time']))
trainer.extend(extensions.PlotReport(
    ['main/loss', 'val/main/loss'],
    x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(
    ['main/accuracy', 'val/main/accuracy'],
    x_key='epoch', file_name='accuracy.png'))

trainer.run()


# When learning is over, save the array containing the state name
# and the learned model as separate files.

import json

with open('state_names.json', 'w') as file_to_save:
    json.dump(state_names, file_to_save)

model.to_cpu()
chainer.serializers.save_npz('laser_machine_listener.model', model)
