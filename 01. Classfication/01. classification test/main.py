"""
Author : Sol Han
re-create Date : 2024.01.23.
"""
from tensorflow import keras
import data_reader

# Epochs
EPOCHS = 20 

# Data Read
dr = data_reader.DataReader()

# Create Nueral network
model = keras.Sequential([
    keras.layers.Dense(3),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(3, activation='softmax')
])

# compile

model.compile(optimizer="adam", metrics=["accuracy"],
              loss="sparse_categorical_crossentropy")

# Train
print("************ TRAINING START ************")
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(dr.train_X, dr.train_Y, epochs=EPOCHS,
                    validation_data=(dr.test_X, dr.test_Y),
                    callbacks=[early_stop])

# Result view
data_reader.draw_graph(history)


# save model
model.save_weights('./my_checkpoint')

# Restore the weights
model.load_weights('./my_checkpoint')

# test
test_data = dr.test_X
test_label = dr.test_Y

loss, acc = model.evaluate(test_data, test_label, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))