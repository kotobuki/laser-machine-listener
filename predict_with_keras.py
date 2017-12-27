import numpy as np
import librosa
import pyaudio
import time
import requests
import json
import keras


IFTTT_EVENT = 'laser_machine_state'
IFTTT_KEY = '*********************'
IFTTT_URL = 'https://maker.ifttt.com/trigger/{event}/with/key/{key}'.format(
    event=IFTTT_EVENT, key=IFTTT_KEY)


def make_web_request(state):
    data = {}
    data['value1'] = state
    try:
        response = requests.post(IFTTT_URL, data=data)
        print('{0.status_code}: {0.text}'.format(response))
    except:
        print('Failed to make a web request')


SAMPLING_RATE = 16000
CHUNK = 1 * SAMPLING_RATE
FFT_SIZE = 256
THRESHOLD = 0.8


with open('state_names.json', 'r') as file_to_load:
    STATES = json.load(file_to_load)
    STATES.append('unknown')

N_STATES = len(STATES) - 1

last_state = STATES.index('unknown')


model = keras.models.load_model('laser_machine_listener.h5')

count = 0
predictions_in_60_sec = np.empty((0, N_STATES))

audio_interface = pyaudio.PyAudio()
audio_stream = audio_interface.open(format=pyaudio.paInt16,
                                    channels=1,
                                    rate=SAMPLING_RATE,
                                    input=True,
                                    frames_per_buffer=CHUNK)


try:
    while True:
        data = np.fromstring(audio_stream.read(CHUNK),
                             dtype=np.int16)

        audio_stream.stop_stream()

        start = time.time()

        state = last_state

        D = librosa.stft(librosa.util.normalize(data),
                         n_fft=FFT_SIZE,
                         window='hamming')
        magnitude = np.abs(D).transpose()

        predictions = model.predict_proba(magnitude, verbose=False)
        predictions_mean = predictions.mean(axis=0)
        predictions_in_60_sec = np.vstack([predictions_in_60_sec, predictions])
        count = count + 1

        elapsed_time = time.time() - start

        localtime = '{0.tm_hour:02d}:{0.tm_min:02d}:{0.tm_sec:02d}'.format(
            time.localtime(time.time()))
        print('{0:s} {1:s} ({2:.3f}, processed in {3:.3f} seconds)'.format(
            localtime,
            STATES[predictions_mean.argmax()],
            predictions_mean.max(),
            elapsed_time))

        if predictions_mean.max() > THRESHOLD:
            state = predictions_mean.argmax()

        if last_state != state:
            print('Changed: {0} > {1}'.format(
                STATES[last_state], STATES[state]))
            last_state = state

        if (count == 60):
            start = time.time()
            state_in_60_sec = predictions_in_60_sec.mean(axis=0).argmax()
            make_web_request(STATES[state_in_60_sec])
            predictions_in_60_sec = np.empty((0, N_STATES))
            count = 0
            elapsed_time = time.time() - start
            print('Made a request in {0:.3f} seconds)'.format(elapsed_time))

        audio_stream.start_stream()

except KeyboardInterrupt:
    print('Requested to terminate')

finally:
    audio_stream.stop_stream()
    audio_stream.close()
    audio_interface.terminate()
    print('Terminated')
