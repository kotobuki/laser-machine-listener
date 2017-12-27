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


# Now the data was prepared as above. First, define a model of
# the neural network with Keras, and divide the data and the correct index
# into 80% for learning and the remaining 20% for verification.
# Next, Select Adam as an optimisation method, set the number of epochs to 20,
# and set an early stopping when there is no change in the correct answer rate,
# and execute learning. When using the files bundled as samples,
# the correct answer rate is about 98%. If you do not get the proper answer rate
# for the examples you prepared, try tuning parameters such as `N_MID_UNITS`
# and `BATCH_SIZE`.

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

N_MID_UNITS = 128

model = Sequential()
model.add(Dense(N_MID_UNITS, activation='sigmoid', input_dim=STFT_MATRIX_SIZE))
model.add(Dense(N_MID_UNITS // 2, activation='sigmoid'))
model.add(Dense(n_states, activation='softmax'))
model.compile(Adam(lr=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


from sklearn.model_selection import train_test_split

index_cat = keras.utils.to_categorical(index)
data_train, data_test, index_train, index_test = train_test_split(
    data, index_cat, test_size=0.2)

from keras.callbacks import EarlyStopping

es_callback = EarlyStopping(monitor='val_acc',
                            patience=2,
                            verbose=True,
                            mode='auto')

BATCH_SIZE = 32

model.fit(data_train, index_train, epochs=20, batch_size=BATCH_SIZE,
          validation_split=0.2, callbacks=[es_callback])


# When learning is over, test with the data and correct answer index
# that have been separated for verification.

model.evaluate(data_test, index_test)


# Finally, save the array containing the state name and the learned model
# as separate files.

import json

with open('state_names.json', 'w') as file_to_save:
    json.dump(state_names, file_to_save)

model.save('laser_machine_listener.h5')
