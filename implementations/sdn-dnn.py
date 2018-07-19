from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from keras.utils import plot_model


# Problems:
# No activations have been mentioned in the paper - using keras defaults
# No loss function mentioned - using keras default



model = Sequential()

model.add(Dense(12, activation='relu', input_dim=6))
model.add(Dense(6, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
plot_model(model, to_file='model-sdn-dnn.png', show_layer_names=True, show_shapes=True)

#model.fit(data, labels, epochs=100, batch_size=10)



# Not sure if this is the right way. It could be that we actually only have 4 layers?
# -> If 4, would that be multiclass with only two classes?!
alt_model = Sequential()
alt_model.add(Dense(12, activation='relu', input_dim=6))
alt_model.add(Dense(6, activation='relu'))
alt_model.add(Dense(3, activation='relu'))
alt_model.add(Dense(2, activation='softmax'))
alt_model.compile(RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

alt_model.summary()
plot_model(alt_model, to_file='model-sdn-dnn-alt.png', show_layer_names=True, show_shapes=True)

#alt_model.fit(data, labels, epochs=100, batch_size=10)
