from model import astromodel1
import numpy as np
import os
from astropy.io import fits

def load_and_preprocess(file_path, files):
    # Loading data using astropy
    desired_shape=(200,3600)
    with fits.open(file_path) as hdul:
        data = hdul[0].data/255  # Access the data from the FITS file
        if data.shape != desired_shape:
            data = None
            return None
        return data
    return None

burst_folder = 'data/bursts'
no_burst_folder = 'data/duds'
desired_shape = (200, 3600)
nTest = 100

burst_files = np.array([os.path.join(burst_folder, file) for file in os.listdir(burst_folder)])
no_burst_files = np.array([os.path.join(no_burst_folder, file) for file in os.listdir(no_burst_folder)])

model = astromodel1()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

predSum = 0
model.load_weights('models/model4645Checkpoint.h5')
for i in range(nTest//2):
    datapoint = None
    while datapoint is None:
        datapoint = load_and_preprocess(np.random.choice(burst_files, size=1)[0], burst_files)
    datapoint = np.array([datapoint[:, :, None]])
    prediction = model.predict(datapoint)
    predSum += prediction[0][0]
    print('burst: ', prediction)
print('avg: ', predSum/(nTest//2))
    
predSum = 0
for i in range(nTest//2): 
    datapoint = None
    while datapoint is None:
        datapoint = load_and_preprocess(np.random.choice(no_burst_files, size=1)[0], no_burst_files)
    datapoint = np.array([datapoint[:, :, None]])
    prediction = model.predict(datapoint)
    predSum += prediction[0][0]
    print('no burst: ', prediction)
print('avg: ', predSum/(nTest//2))