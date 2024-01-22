from keras import backend as K
from model import *
import numpy as np
import os
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from astropy.utils.data import get_pkg_data_filename
from astropy.visualization import astropy_mpl_style
from matplotlib.patches import Rectangle
import scipy.stats

from matplotlib.widgets import Button

plt.style.use(astropy_mpl_style)

def load_and_preprocess(file_path):
    desired_shape=(200,3600)
    with fits.open(file_path) as hdul:
        data = None
        data = hdul[0].data/255 
        if data.shape != desired_shape:
            data = None
        return data
    return None

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

model = astromodel3()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.load_weights("models/model7Checkpoint.h5")
    
burst_folder = 'data\\bursts'
# no_burst_folder = 'data\\duds'

desired_shape = (200, 3600)

burst_files = np.array([os.path.join(burst_folder, file) for file in os.listdir(burst_folder)])
# no_burst_files = np.array([os.path.join(no_burst_folder, file) for file in os.listdir(no_burst_folder)])

IQRthreshold = 1.5

last_layer_channels = 128
last_layer_size = 130

model_size = 14

# print(model.weights)

while True:
    datapoint = None
    while datapoint is None:
        file = np.random.choice(burst_files, size=1)[0]
        datapoint = load_and_preprocess(file)
    datapoint = np.array([datapoint[:, :, None]])

    print(file)

    get_3rd_layer_output = K.function([model.layers[0].input],
                                    [model.layers[model_size-2].output])
    layer_output = get_3rd_layer_output(datapoint)[0]

    weights = model.layers[model_size].get_weights()[0].reshape(last_layer_size, last_layer_channels)
    averageweights = np.array([np.average(i) for i in weights.T])
    layer_output = layer_output.reshape((last_layer_size, last_layer_channels))
    # output = np.dot(layer_output, weights.T)
    # output = np.array([np.sum(i) for i in output])
    output = np.array([np.sum(i*averageweights) for i in layer_output])

    # output = model.layers[model_size].get_weights()[0].T * layer_output
    # output = np.array([np.sum(i) for i in output.reshape(last_layer_size, last_layer_channels)])

    # smoothoutput = moving_average(output, 10)

    # q75, q50, q25 = np.percentile(output, [75 ,50, 25])
    # iqr = q75 - q25

    predictedvalue = model.predict(datapoint)[0][0]

    fig, ax = plt.subplots()

    x = []
    y = []
    step = desired_shape[1] / last_layer_size
    for i in output:
        x.append(len(x)*step)
        y.append(i*100)
        # if(abs(i-q50)>(iqr*IQRthreshold)):
        # v = scipy.stats.percentileofscore(output, i)/100
        # ax.add_patch(Rectangle((x[-1], 0), step, desired_shape[0], facecolor=((1-v),v,0,0.2)))
        # ax.add_patch(Rectangle((x[-1], 0), step, desired_shape[0], facecolor=(0,1,0,max(i,0))))

    image_file = get_pkg_data_filename(file)
    image_data = fits.getdata(image_file, ext=0)

    plt.title("Prediction: {:0.4f}".format(predictedvalue))
    plt.suptitle(file)
    plt.imshow(image_data, cmap='gray', interpolation='nearest', aspect='auto')
    plt.scatter(x,y)
    plt.show()