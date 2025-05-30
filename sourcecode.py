import pandas as pd
import numpy as np

np.random.seed(1212)

import keras
from keras.models import Model
from keras.layers import *
from keras import optimizers
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_train.head() # 784 features, 1 label
df_features = df_train.iloc[:, 1:785]
df_label = df_train.iloc[:, 0]

X_test = df_test.iloc[:, 0:784]

print(X_test.shape)
from sklearn.model_selection import train_test_split

# Split the dataset into training and cross-validation sets
X_train, X_cv, y_train, y_cv = train_test_split(
    df_features, df_label,
    test_size=0.2,
    random_state=1212
)

# Convert DataFrames to NumPy arrays and reshape
X_train = X_train.values.reshape(-1, 784)  # Reshape automatically to (33600, 784)
X_cv = X_cv.values.reshape(-1, 784)        # Reshape to (8400, 784)
X_test = X_test.values.reshape(-1, 784)    # Reshape to (28000, 784)
print((min(X_train[1]), max(X_train[1])))
# Feature Normalization
X_train = X_train.astype('float32'); X_cv= X_cv.astype('float32'); X_test = X_test.astype('float32')
X_train /= 255; X_cv /= 255; X_test /= 255

# Convert labels to One Hot Encoded
num_digits = 10
y_train = keras.utils.to_categorical(y_train, num_digits)
y_cv = keras.utils.to_categorical(y_cv, num_digits)
print((min(X_train[1]), max(X_train[1])))
# Input Parameters
n_input = 784 # number of features
n_hidden_1 = 300
n_hidden_2 = 100
n_hidden_3 = 100
n_hidden_4 = 200
num_digits = 10
Inp = Input(shape=(784,))
x = Dense(n_hidden_1, activation='relu', name = "Hidden_Layer_1")(Inp)
x = Dense(n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)
x = Dense(n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)
x = Dense(n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)
output = Dense(num_digits, activation='softmax', name = "Output_Layer")(x)
# Our model would have '6' layers - input layer, 4 hidden layer and 1 output layer
model = Model(Inp, output)
model.summary() # We have 297,910 parameters to estimate
from tensorflow.keras import optimizers

# Insert Hyperparameters
learning_rate = 0.1
training_epochs = 20
batch_size = 100

# Initialize SGD optimizer
sgd = optimizers.SGD(learning_rate=learning_rate)
# We rely on the plain vanilla Stochastic Gradient Descent as our optimizing methodology
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
history1 = model.fit(X_train, y_train,
                     batch_size = batch_size,
                     epochs = training_epochs,
                     verbose = 2,
                     validation_data=(X_cv, y_cv))


# Hyperparameters
learning_rate = 0.001  # Adam usually works better with smaller learning rates
n_hidden_1 = 128
n_hidden_2 = 64
n_hidden_3 = 64
n_hidden_4 = 32
num_digits = 10  # For MNIST

# Build model
Inp = Input(shape=(784,))
x = Dense(n_hidden_1, activation='relu', name="Hidden_Layer_1")(Inp)
x = Dense(n_hidden_2, activation='relu', name="Hidden_Layer_2")(x)
x = Dense(n_hidden_3, activation='relu', name="Hidden_Layer_3")(x)
x = Dense(n_hidden_4, activation='relu', name="Hidden_Layer_4")(x)
output = Dense(num_digits, activation='softmax', name="Output_Layer")(x)

# Create and compile model
adam = Adam(learning_rate=learning_rate)
model2 = Model(Inp, output)

model2.compile(
    loss='categorical_crossentropy',
    optimizer=adam,
    metrics=['accuracy']
)
history2 = model2.fit(X_train, y_train,
                      batch_size = batch_size,
                      epochs = training_epochs,
                      verbose = 2,
                      validation_data=(X_cv, y_cv))


# Define the model architecture
Inp = Input(shape=(784,))
x = Dense(n_hidden_1, activation='relu', name="Hidden_Layer_1")(Inp)
x = Dense(n_hidden_2, activation='relu', name="Hidden_Layer_2")(x)
x = Dense(n_hidden_3, activation='relu', name="Hidden_Layer_3")(x)
x = Dense(n_hidden_4, activation='relu', name="Hidden_Layer_4")(x)
output = Dense(num_digits, activation='softmax', name="Output_Layer")(x)

# Instantiate optimizer
adam = Adam(learning_rate=learning_rate)

# Build and compile the model
model2a = Model(Inp, output)
model2a.compile(loss='categorical_crossentropy',
                optimizer=adam,
                metrics=['accuracy'])


# Build model
Inp = Input(shape=(784,))
x = Dense(n_hidden_1, activation='relu', name="Hidden_Layer_1")(Inp)
x = Dense(n_hidden_2, activation='relu', name="Hidden_Layer_2")(x)
x = Dense(n_hidden_3, activation='relu', name="Hidden_Layer_3")(x)
x = Dense(n_hidden_4, activation='relu', name="Hidden_Layer_4")(x)
output = Dense(num_digits, activation='softmax', name="Output_Layer")(x)

# Optimizer
adam = Adam(learning_rate=learning_rate)

# Model
model2b = Model(Inp, output)
model2b.compile(
    loss='categorical_crossentropy',
    optimizer=adam,  # use the Adam instance
    metrics=['accuracy']
)
history2b = model2b.fit(X_train, y_train,
                        batch_size = batch_size,
                        epochs = training_epochs,
                            validation_data=(X_cv, y_cv))
# Input Parameters
n_input = 784 # number of features
n_hidden_1 = 300
n_hidden_2 = 100
n_hidden_3 = 100
n_hidden_4 = 100
n_hidden_5 = 200
num_digits = 10
Inp = Input(shape=(784,))
x = Dense(n_hidden_1, activation='relu', name = "Hidden_Layer_1")(Inp)
x = Dense(n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)
x = Dense(n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)
x = Dense(n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)
x = Dense(n_hidden_5, activation='relu', name = "Hidden_Layer_5")(x)
output = Dense(num_digits, activation='softmax', name = "Output_Layer")(x)
# Our model would have '7' layers - input layer, 5 hidden layer and 1 output layer
model3 = Model(Inp, output)
model3.summary() # We have 308,010 parameters to estimate
from tensorflow.keras.optimizers import Adam

# Define Adam optimizer with a specific learning rate
adam = Adam(learning_rate=0.01)

# Compile the model using the Adam instance
model3.compile(
    loss='categorical_crossentropy',
    optimizer=adam,
    metrics=['accuracy']
)
history3 = model3.fit(X_train, y_train,
                      batch_size = batch_size,
                      epochs = training_epochs,
                      validation_data=(X_cv, y_cv))
                      # Input Parameters
n_input = 784 # number of features
n_hidden_1 = 300
n_hidden_2 = 100
n_hidden_3 = 100
n_hidden_4 = 200
num_digits = 10
Inp = Input(shape=(784,))
x = Dense(n_hidden_1, activation='relu', name = "Hidden_Layer_1")(Inp)
x = Dropout(0.3)(x)
x = Dense(n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)
x = Dropout(0.3)(x)
x = Dense(n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)
x = Dropout(0.3)(x)
x = Dense(n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)
output = Dense(num_digits, activation='softmax', name = "Output_Layer")(x)
# Our model would have '6' layers - input layer, 4 hidden layer and 1 output layer
model4 = Model(Inp, output)
model4.summary() # We have 297,910 parameters to estimate
model4.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
test_pred = pd.DataFrame(model4.predict(X_test, batch_size=200))
test_pred = pd.DataFrame(test_pred.idxmax(axis = 1))
test_pred.index.name = 'ImageId'
test_pred = test_pred.rename(columns = {0: 'Label'}).reset_index()
test_pred['ImageId'] = test_pred['ImageId'] + 1

test_pred.head()
test_pred.to_csv('mnist_submission.csv', index = False)
         


