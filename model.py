from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

def astromodel1():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=(200, 3600, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3),  activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3),  activation='relu'))
    model.add(MaxPooling2D(pool_size=(12, 3)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

def astromodel2():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(200, 3600, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3),  activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3),  activation='relu'))
    model.add(MaxPooling2D(pool_size=(12, 3)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

def astromodel3():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=(200, 3600, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3),  activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3),  activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3),  activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 1)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

if __name__ == '__init__':
    model=astromodel2()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()