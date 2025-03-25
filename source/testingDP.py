import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from time import perf_counter as t
import sys
import json
from matplotlib.lines import Line2D
from tensorflow_privacy import DPKerasSGDOptimizer
#Evaluation: evaluate the accuracy and loss of IPPML
#build and train the benchmark
#privacy preserving tensorflow example: https://www.tensorflow.org/responsible_ai/privacy/tutorials/classification_privacy
def build_benchmark(ds_name, ds_info, withDP):

  learning_rate = 0.4 #, 'Learning rate for training')
  noise_multiplier = 0.1 #'Ratio of the standard deviation to the clipping norm')
  l2_norm_clip = 1.0 #, 'Clipping norm')
  microbatches = 2 #, 'Number of microbatches '(must evenly divide batch_size)')

  
  opt = tf.keras.optimizers.Adam(0.001)
  los = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)
  if withDP:
    opt = DPKerasSGDOptimizer(l2_norm_clip= l2_norm_clip, noise_multiplier= noise_multiplier, num_microbatches= microbatches, learning_rate= learning_rate)
    los = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)

  benchmark_ds_train = tfds.load(
      ds_name, 
      split = 'train',
      as_supervised=True
    )
  benchmark_ds_train = benchmark_ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
  benchmark_ds_train = benchmark_ds_train.cache()
  benchmark_ds_train = benchmark_ds_train.shuffle(ds_info.splits['train'].num_examples)
  benchmark_ds_train = benchmark_ds_train.batch(128)
  benchmark_ds_train = benchmark_ds_train.prefetch(tf.data.AUTOTUNE)

  benchmark_model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='sigmoid'),
      tf.keras.layers.Dense(200, activation='sigmoid'),
      tf.keras.layers.Dense(128, activation='sigmoid'),
      tf.keras.layers.Dense(10)
    ])


  benchmark_model.compile(
      optimizer = opt,
      # Compute vector of per-example loss rather than its mean over a minibatch.
      loss = los,
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
  benchmark_model._name = "benchmark_model"
  if withDP:
    benchmark_model._name = "benchmark_withDP_model"

  return benchmark_model, benchmark_ds_train
  
def train_benchmark(benchmark_model, benchmark_ds_train):
  benchmark_model.fit(
          benchmark_ds_train,
          epochs=6,
          verbose = 0
      )
  return benchmark_model

  # build a training pipeline and pass each segment through it
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


# build an evaluation pipeline
def test_pipeline(ds_test):
  ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
  ds_test = ds_test.batch(128)
  ds_test = ds_test.cache()
  ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

  return ds_test



ds_name = 'mnist'
ds_test, ds_info = tfds.load(
      ds_name, 
      split='test',
      shuffle_files=True,
      as_supervised=True,
      with_info=True
  )
ds_test = test_pipeline(ds_test) #pipeline for the test data

benchmark_model, benchmark_ds_train = build_benchmark(ds_name, ds_info, withDP = True)

# START latency of **training** & **retraining** benchmark method
start_benchmark_training = t()
start_benchmark_retraining = t()
benchmark_model = train_benchmark(benchmark_model, benchmark_ds_train)
# END latency of **training** & **retraining** benchmark method
latency_benchmark_training = t() - start_benchmark_training
latency_benchmark_retraining = t() - start_benchmark_retraining

# START latency of **prediction**
start_benchmark_prediction = t()
loss, acc = benchmark_model.evaluate(ds_test)
# END latency of **prediction** (/len(ds_test))
latency_benchmark_prediction = (t() - start_benchmark_prediction)/len(ds_test)

acc_loss_dict = dict() #the key is '1' because the benchmark has one segment
acc_loss_dict['acc'] = round(acc, 3)
acc_loss_dict['loss'] = round(loss, 3)

acc_loss_dict['training'] = latency_benchmark_training
acc_loss_dict['retraining'] = latency_benchmark_retraining
acc_loss_dict['prediction'] = latency_benchmark_prediction






	# tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3))
	# tf.keras.layers.BatchNormalization()
	# tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')
	# tf.keras.layers.BatchNormalization()
	# tf.keras.layers.MaxPooling2D((2, 2))
	# tf.keras.layers.Dropout(0.2)
	# tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')
	# tf.keras.layers.BatchNormalization()
	# tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')
	# tf.keras.layers.BatchNormalization()
	# tf.keras.layers.MaxPooling2D((2, 2))
	# tf.keras.layers.Dropout(0.3)
 

	# tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')
	# tf.keras.layers.BatchNormalization()
	# tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')
	# tf.keras.layers.BatchNormalization()
	# tf.keras.layers.MaxPooling2D((2, 2))
	# tf.keras.layers.Dropout(0.4)
	# tf.keras.layers.Flatten()
	# tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform')
	# tf.keras.layers.BatchNormalization()
	# tf.keras.layers.Dropout(0.5)
	# tf.keras.layers.Dense(10, activation='softmax')