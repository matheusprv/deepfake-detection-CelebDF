import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping

class SaveEpochsInfo(tf.keras.callbacks.Callback):
    def __init__(self, file):
        self.save_test_results = False
        self.file = file

    def on_epoch_end(self, epoch, logs=None):
        with open(self.file, "a") as f:
            text = f"epoch: {epoch + 1} "

            for key in logs.keys():
                text += f"{key}: {logs[key]} "

            f.write(text + "\n")

    def on_test_end(self, logs=None):
        if self.save_test_results:
            self.on_epoch_end(-1, logs)

    def on_train_end(self, logs=None):
        self.save_test_results = True


def model_training_and_evaluation(model, train_generator, val_generator, test_generator, save_best_file_name, save_training_file_name):

    save_best = tf.keras.callbacks.ModelCheckpoint(filepath=save_best_file_name, monitor='val_loss', save_best_only=True, mode='min')
#    early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode="min", restore_best_weights=True)
    save_epochs = SaveEpochsInfo(save_training_file_name)

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate =   0.01,
        decay_steps           =   train_generator.n,
        decay_rate            =   0.5,
        staircase=True)
    optimizer= SGD(learning_rate = learning_rate, momentum = 0.9)
    loss = "categorical_crossentropy"
    model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy', 'AUC'])

    history = model.fit(
        train_generator,
        epochs = 15,
        shuffle = True,
        validation_data = val_generator,
        callbacks=[save_best, save_epochs]
    )


    model.evaluate(test_generator, callbacks = [save_epochs])
    model.save("Train_saved.keras")

    model.load_weights(save_best_file_name)
    model.evaluate(test_generator, callbacks = [save_epochs])
