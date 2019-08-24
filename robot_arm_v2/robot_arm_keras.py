'''
import tensorflow as tf
import numpy as np

# Read in training data
x_train = np.load('data/position_train_120k.npy').astype('float32')
y_train = np.load('data/joint_train_120k.npy').astype('float32')
assert (x_train.shape[0] == y_train.shape[0])
# Read in testing data
x_test = np.load('data/position_test_29k.npy').astype('float32')
y_test = np.load('data/joint_test_29k.npy').astype('float32')
assert (x_test.shape[0] == y_test.shape[0])

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1024, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(256, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(6, activation=None)
])
model.compile(optimizer='Adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
score = model.evaluate(x_test, y_test)

print('Test loss: {}'.format(score[0]))
print('Test accuracy: {}'.format(score[1]))


# predicting the angle (in radians)
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
# Read in training data
def standardization(data):
    return ((data.T - np.mean(data, axis=1)) / np.std(data, axis=1)).T
X = standardization(np.load('data/position_train_120k.npy').astype('float32'))
y = np.load('data/joint_train_120k.npy').astype('float32')
size = len(X)
assert (X.shape[0] == y.shape[0])
# generate toy data
np.random.seed(1)
# simple prediction
model = MLPRegressor(random_state=42, activation='tanh', max_iter=10000)
y_simple_pred = cross_val_predict(model, X, y)
# transformed prediction
joint = cross_val_predict(model, X, np.column_stack([np.sin(y), np.cos(y)]))
print(joint.shape)
y_trig_pred = np.arctan2(joint[:,0], joint[:,1]).reshape(size, 1)
i = 2
for _ in range(5):
    y_trig_pred = np.append(y_trig_pred, np.arctan2(joint[:,i], joint[:,i+1]).reshape(size,1), axis=1)
    i+=2
print(y_trig_pred.shape)
# compare
def align(y_true, y_pred):
    """ Add or remove 2*pi to predicted angle to minimize difference from GT"""
    y_pred = y_pred.copy()
    y_pred[y_true-y_pred >  np.pi] += np.pi*2
    y_pred[y_true-y_pred < -np.pi] -= np.pi*2
    return y_pred
print(type(y_simple_pred))
print(type(y_trig_pred))
print(r2_score(y, align(y, y_simple_pred))) # R^2 about 0.57
print(r2_score(y, align(y, y_trig_pred)))   # R^2 about 0.99
plt.figure(figsize=(12, 3))
plt.subplot(3,3,1)
plt.scatter(X[:,0], X[:,1])
plt.title('Data (y=color)'); plt.xlabel('x1'); plt.ylabel('x2')
plt.subplot(3,3,2)
plt.scatter(y_simple_pred, y)
plt.title('Direct model'); plt.xlabel('prediction'); plt.ylabel('actual')
plt.subplot(3,3,3)
plt.scatter(y_trig_pred, y)
plt.title('Sine-cosine model'); plt.xlabel('prediction'); plt.ylabel('actual')
plt.subplot(3,3,4)
plt.scatter(joint[:,0], joint[:,1], s=5)
plt.title('Predicted sin and cos'); plt.xlabel('cos'); plt.ylabel('sin')
plt.subplot(3,3,5)
plt.scatter(joint[:,2], joint[:,3], s=5)
plt.title('Predicted sin and cos'); plt.xlabel('cos'); plt.ylabel('sin')
plt.subplot(3,3,6)
plt.scatter(joint[:,4], joint[:,5], s=5)
plt.title('Predicted sin and cos'); plt.xlabel('cos'); plt.ylabel('sin')
plt.subplot(3,3,7)
plt.scatter(joint[:,6], joint[:,7], s=5)
plt.title('Predicted sin and cos'); plt.xlabel('cos'); plt.ylabel('sin')
plt.subplot(3,3,8)
plt.scatter(joint[:,8], joint[:,9], s=5)
plt.title('Predicted sin and cos'); plt.xlabel('cos'); plt.ylabel('sin')
plt.subplot(3,3,9)
plt.scatter(joint[:,10], joint[:,11], s=5)
plt.title('Predicted sin and cos'); plt.xlabel('cos'); plt.ylabel('sin')
plt.tight_layout();
plt.show()
'''

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
import numpy
from keras import backend as K
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import numpy as np
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
def standardization(data):
    return ((data.T - np.mean(data, axis=1)) / np.std(data, axis=1)).T
# load dataset with position,orientation and joint angles
X = np.round(np.load('data/position_train_120k.npy').astype('float32'), decimals=3)
Y = np.round(np.load('data/joint_train_120k.npy').astype('float32'), decimals=3)
X_test = np.round(np.load('data/position_test_29k.npy').astype('float32'), decimals=3)
Y_test = np.round(np.load('data/joint_test_29k.npy').astype('float32'), decimals=3)
print(X.shape,Y.shape)

def dist(y_true, y_pred):
    return tf.linalg.norm(y_true-y_pred)

##define base model
def base_model():
     model = Sequential()
     model.add(Dense(512, input_dim=6, activation='softsign'))
     model.add(Dropout(0.2))
     model.add(Dense(1024, activation='softsign'))
     model.add(Dropout(0.2))
     model.add(Dense(512, activation='softsign'))
     model.add(Dropout(0.2))
     model.add(Dense(6, init='normal'))
     model.compile(loss='mean_squared_error', 
                   optimizer='rmsprop',
                   metrics=['mae', 'mse','acc', dist])
     return model

def base_model1():
     model = Sequential()
     model.add(Dense(7, input_dim=6, init='normal', activation='tanh'))
     model.add(Dense(7, init='normal', activation='tanh'))
     model.add(Dense(7, init='normal', activation='tanh'))
     model.add(Dense(6, init='normal'))
     model.compile(loss='mean_squared_error', 
                   optimizer='rmsprop',
                   metrics=['mae', 'mse','acc', dist])
     return model


clf = KerasRegressor(build_fn=base_model, epochs=10000, batch_size=120000,verbose=2)

clf.fit(X,Y)
res = clf.predict(X_test)
print(res)


mse = np.sqrt(mean_squared_error(Y_test, res))
print(mse)
# ... code
K.clear_session()

