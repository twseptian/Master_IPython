from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from keras.utils import plot_model

batch_size =    10
epochs     =    100
learn_rate =    0.001

model = Sequential()

model.add(Dense(12, activation='relu', input_dim=6))
model.add(Dense(6, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.compile(RMSprop(lr=learn_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()
plot_model(model, to_file='model-sdn-dnn.png', show_layer_names=True, show_shapes=True)

history = model.fit(kdd_train, kdd_train_labels, 
                    epochs=epochs, 
                    batch_size=batch_size,
                    verbose=1,
                    validation_data=(kdd_test, kdd_test_labels),
                    callbacks=callbacks)