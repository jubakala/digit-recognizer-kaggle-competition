import pandas as pd
import numpy as np
import tensorflow as tf

"""
This is a simple convolutional neural network I used to participate Kaggle's Digit Recognizer competition (https://www.kaggle.com/c/digit-recognizer).
With the settings above, I was able to reach the accuracy of 0.99071.
I also test following settings:
    - Dropout 0.25 instead of 0.3
    - Batch size of 64 instead of 32
    - Epoch count of 25 and 30, instead of 30.
    - An extra set of Conv2D, Maxpooling and Dropout -layers, identical to the set in the middle of the model.

The settings listed above resulted great results (0.981, 0.9904 & 0.99007) as well, but not as good as the best configuration.
"""

def get_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    IMG_SIZE = 28

    # Preparing the dataset from csv-rows to images.
    X_train = train.iloc[:,1:]
    X_train = X_train.values.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_train = train.iloc[:,0]
    X_test = test
    X_test = X_test.values.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, X_test, y_train, test


def get_model(input_shape):
    # Constructing a simple convolutional neural network.
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))    

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

    return model


def create_result_file(predictions, test, filename):
    solutions = []

    for i in range(len(predictions)):
        solutions.append(np.argmax(predictions[i]))

    final = pd.DataFrame()
    final['ImageId']=[i+1 for i in test.index]
    final['Label']=solutions
    final.to_csv("results/" + filename, index=False)


def main():
    EPOCHS = 20

    # Get the data.
    X_train, X_test, y_train, test = get_data()
    # Create the model.
    model = get_model(X_train.shape[1:])
    
    # Fit the model.
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, validation_split=0.2)

    # Make predictions on the test data.
    predictions = model.predict([X_test])
    # Save the predictions to a file.
    create_result_file(predictions, test, "cnn_results.csv")


if __name__ == "__main__":
    main()