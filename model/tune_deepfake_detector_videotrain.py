FRAMES_PER_VIDEO = 24
BATCH_SIZE = 2
HEIGHT = WIDTH = 299

input_shape = (FRAMES_PER_VIDEO, HEIGHT, WIDTH, 3)

colab = False
root_folder = None
if colab:
    root_folder = "/content/Celeb-df-V2-faces-extracted"
else:
    root_folder = "/media/work/matheusvieira/deep_fake_detection/Celeb-df-V2-faces-extracted/"

#######################################################################################################

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling3D, Input, TimeDistributed, BatchNormalization
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.optimizers import SGD

import tensorflow as tf
import os
from VideoDataGenerator import CustomDataGen
from Training import model_training_and_evaluation

#######################################################################################################

train_generator = CustomDataGen(os.path.join(root_folder, 'train'), BATCH_SIZE, will_manipulate_video = False)
val_generator = CustomDataGen(os.path.join(root_folder, 'val'), BATCH_SIZE, will_manipulate_video = False)
test_generator = CustomDataGen(os.path.join(root_folder, 'test'), BATCH_SIZE, will_manipulate_video = False)

print("train:",train_generator.n)
print("val:",val_generator.n)
print("test:",test_generator.n)

#######################################################################################################

def build_model(hp):
    base_model = Xception(include_top=False, weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = True

    x = input_layer = Input(shape=input_shape)
    x = TimeDistributed(base_model)(x)
    x = GlobalAveragePooling3D()(x)
    x = Dense(hp.Int(("dense_1"), min_value = 256, max_value = 400, step = 16), activation = 'relu')(x)
    #x = Dense(hp.Int(("dense_2"), min_value = 32, max_value = 128, step = 16), activation = 'relu')(x)
    output = Dense(2, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output)

    # Optimizer
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    momentum = hp.Float('momentum', min_value= 0.87, max_value=0.92, sampling='log')
    nesterov = hp.Boolean('nesterov')
    optimizer = SGD(learning_rate = learning_rate, momentum = momentum, nesterov = nesterov)
    
    # Compile model
    model.compile(
        optimizer = optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'AUC']
    )

    return model

#######################################################################################################
# Tuning the model
import keras_tuner as kt

tuner = kt.BayesianOptimization(
    build_model,
    objective    = 'val_loss',
    max_trials   = 20,
    directory    = '/media/work/matheusvieira/',
    project_name = 'tune_deepfake_detector'
)

tuneEarlyStopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

tuner.search(
	train_generator,
	validation_data = val_generator,
	epochs = 7,
	callbacks = [tuneEarlyStopping]
)

#######################################################################################################

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
dicionario = best_hps.get_config()["values"]

with open('hyperparameters.txt', 'w') as file:
    # Iterate over dictionary items
    for key, value in dicionario.items():
        # Write key-value pair to the file
        file.write(f"{key}: {value}\n")

model = tuner.hypermodel.build(best_hps)

#######################################################################################################
# Training the tunned model
model_training_and_evaluation(
    model, 
    train_generator, 
    val_generator, 
    test_generator, 
    save_best_file_name = "cnn_fully.keras",
    save_training_file_name = "cnn_fully_history.txt",
    compile=False
)