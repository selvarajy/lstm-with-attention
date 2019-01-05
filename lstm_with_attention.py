import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Input, LSTM, multiply,Bidirectional
from keras.models import Model
from sklearn import preprocessing
from keras.callbacks import Callback
from keras.layers.core import *
from keras.optimizers import *

time_steps = 3
features = 1
column = 'A'
batch_size = 1
stateful = True
single = False

# callback to reset states
class ResetCallback(Callback):
    def __init__(self):
        self.counter = 0

    def on_batch_begin(self, batch, logs={}):
        if stateful:
            if self.counter % time_steps == 0:
                self.model.reset_states()
        self.counter += 1
        
def create_dataset(data, time_steps = 1):
    X,Y = [],[]
    for i in range(len(data)-time_steps-1):
        X.append(data[i:(i+time_steps),0])
        Y.append(data[(i+time_steps),0])
    return np.array(X),np.array(Y)

data = pd.read_excel('data.xlsx')

series = data[column].to_frame().values.astype('float32')
series = preprocessing.scale(series)

#split data
train_size = int(len(series)*0.8)
train = series[0:train_size]
test = series[train_size:]
trainX,trainY = create_dataset(train,time_steps)
testX,testY = create_dataset(test, time_steps)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], features))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], features))

def attention_lstm(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    print(inputs.shape)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, time_steps))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(time_steps, activation='softmax')(a)
    if single:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    attention_m = multiply([inputs, a_probs])
    return attention_m

#network
input = Input(batch_shape=(batch_size,time_steps,features), name='input', dtype='float32')
lstm_layer1 = Bidirectional(LSTM(units=4*int(time_steps*2/3+1), return_sequences=True, return_state=False, stateful=stateful)) (input)
attention_m = attention_lstm(lstm_layer1)
attention_m = Flatten()(attention_m)   
output = Dense(units=1) (attention_m)

#optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model = Model(inputs = input, outputs = output)
model.compile(optimizer= optimizer, loss='mean_squared_error', metrics=['accuracy'])
history = model.fit(trainX, trainY, batch_size=batch_size, epochs=4000, verbose=2, validation_split=0.2, shuffle=not(stateful), callbacks=[ResetCallback()])

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


#######################################################

# generate predictions for training
trainPredict = model.predict(trainX, batch_size)
model.reset_states()
testPredict = model.predict(testX, batch_size)


trainPred = pd.Series(trainPredict[:,0])
trainAct = pd.Series(trainY)

# shift train predictions for plotting
trainPredictPlot = np.empty_like(train)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_steps:len(trainPredict)+time_steps, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(series)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(time_steps*2)+1:len(series)-1, :] = testPredict
# plot baseline and predictions
plt.plot(series)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
