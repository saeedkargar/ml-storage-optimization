
# Tensorflow datasets: https://www.tensorflow.org/datasets/catalog/overview
# Tensorflow CNNs: https://www.tensorflow.org/api_docs/python/tf/keras/applications
#reference: https://www.tensorflow.org/datasets/keras_example
#Keras documentation: https://keras.io/search.html
# M1 chip tensorflow: https://developer.apple.com/metal/tensorflow-plugin/

import tensorflow as tf
import tensorflow_datasets as tfds
import os
from time import perf_counter as t
import json
from tensorflow_privacy import DPKerasSGDOptimizer
from tensorflow_privacy import DPKerasAdamOptimizer
# import tensorflow_addons as tfa
#========================================================================
# import tensorflow_privacy
# from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
import numpy as np
tf.compat.v1.disable_v2_behavior()
tf.get_logger().setLevel('ERROR')
#========================================================================
# from playsound import playsound
# from logging import info, error, debug 
# from keras.utils.vis_utils import plot_model

from math import ceil

batch_size = 250

basic_mnist_benchmark_model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='sigmoid'),
        tf.keras.layers.Dense(200, activation='sigmoid'),
        tf.keras.layers.Dense(128, activation='sigmoid'),
        tf.keras.layers.Dense(10)
])

vgg_benchmark_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
])

resnet_benchmark_model= tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax') 
])


# basic_mnist_hush_dp_model = tf.keras.models.load_model('/Users/samaagazzaz/Desktop/UCSC_Courses/IPPML/experiments/source/hush_dp/mnist/25-04-2023_22:19:48.h5', compile=False)
# basic_mnist_hush_dp_model = basic_mnist_benchmark_model

def avg(lst):
  return sum(lst) / len(lst)

def load_data(segment_count, segment_range, ds_name):
  train_splits = tfds.even_splits('train', n=segment_count)
  ds_train = list(segment_range)

  for i in segment_range:
    ds_train[i] = tfds.load(
      ds_name, 
      split=train_splits[i],
      as_supervised=True,
    )
    x=0
    for image, label in tfds.as_numpy(ds_train[i]):
      # print(type(image), type(label), label)
      x+=1

    # print(data.shape, len(label))
    print('training segment ', i+1, 'size', x)
    # print('training segment ', i+1, 'size', len(ds_train[i]))
      
    
  return ds_train

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

def segment_train_pipeline(segment_range, ds_train):
  for i in segment_range:
    ds_train[i] = ds_train[i].map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train[i] = ds_train[i].cache()

    x=0
    for image, label in tfds.as_numpy(ds_train[i]):
      # print(type(image), type(label), label)
      x+=1

    ds_train[i] = ds_train[i].shuffle(x)
    ds_train[i] = ds_train[i].batch(batch_size)
    ds_train[i] = ds_train[i].prefetch(tf.data.AUTOTUNE)
  return ds_train

def test_pipeline(ds_test):
  ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
  ds_test = ds_test.batch(batch_size)
  ds_test = ds_test.cache()
  ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

  return ds_test

def build_root_iverted(model_name, percent):
  if model_name == 'mnist-basic':
    if percent == 0.5:
      root_model = tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
          tf.keras.layers.Dense(128, activation='sigmoid'),
          tf.keras.layers.Dense(200, activation='sigmoid')
          ])

    elif percent == 0.25:
      root_model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='sigmoid')
        ])

    elif percent == 0.75:
      root_model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='sigmoid'),
        tf.keras.layers.Dense(200, activation='sigmoid'),
        tf.keras.layers.Dense(128, activation='sigmoid')
        ])

  elif model_name == 'vgg':
    temp_model = tf.keras.models.clone_model(vgg_benchmark_model)
    layer_count = ceil(len(temp_model.layers)*percent)
    print('the root layer count is : ', layer_count)
    if layer_count < 1 or layer_count == len(vgg_benchmark_model.layers):
      print('there was a problem building the root model')
      return None
    root_model = tf.keras.models.Sequential(temp_model.layers[:layer_count])

  elif model_name == 'resnet':
    temp_model = tf.keras.models.clone_model(resnet_benchmark_model)
    layer_count = ceil(len(temp_model.layers)*percent)
    print('the root layer count is : ', layer_count)
    if layer_count < 1  or layer_count == len(resnet_benchmark_model.layers): 
      print('there was a problem building the root model')
      return None
    root_model = tf.keras.models.Sequential(temp_model.layers[:layer_count])

  else:
    print('there was a problem building the root model, model name is not recognized')
    exit()
  
  
  return root_model

def get_optimizer(choice):
  learning_rate = 0.25 #, 'Learning rate for training')
  noise_multiplier = 0.1 #'Ratio of the standard deviation to the clipping norm')
  l2_norm_clip = 1.5 #, 'Clipping norm')
  microbatches = batch_size #, 'Number of microbatches '(must evenly divide batch_size)')

  if choice == "normal":
    return tf.keras.optimizers.Adam(0.001)
  elif choice == "DP":
    return DPKerasAdamOptimizer(l2_norm_clip= l2_norm_clip, noise_multiplier= noise_multiplier, num_microbatches= microbatches, learning_rate= learning_rate)
    
  # elif choice == "mixed":
  #   optimizers = [
  #       tf.keras.optimizers.Adam(0.001),
  #       DPKerasAdamOptimizer(l2_norm_clip= l2_norm_clip, noise_multiplier= noise_multiplier, num_microbatches= microbatches, learning_rate= learning_rate)
  #   ]

  #   optimizers_and_layers = [(optimizers[0], model.layers[0:trunk_count]),
  #                           (optimizers[1], model.layers[trunk_count:])]

  #   return tfa.optimizers.MultiOptimizer(optimizers_and_layers)

def build_segment_models_inverted(model_name, segment_range, root_model, percent):
  if model_name == 'mnist-basic':
    if percent == 0.5:
      #model_segment_full items represent the entirity of the model
      model_segment_full = list(segment_range)
      
      for i in segment_range:
        print('building model_segment_full[', i, '] using data segment', i)
        model_segment_full[i] = tf.keras.models.Sequential([
          root_model,
          tf.keras.layers.Dense(128, activation='sigmoid'),
          tf.keras.layers.Dense(10)
          ])

        model_segment_full[i]._name = "model_full_segment"+str(i)

        model_segment_full[i].compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

    elif percent == 0.25:
      #model_segment_full items represent the entirity of the model
      model_segment_full = list(segment_range)
      
      for i in segment_range:
        print('building model_segment_full[', i, '] using data segment', i)
        model_segment_full[i] = tf.keras.models.Sequential([
          root_model,
          tf.keras.layers.Dense(200, activation='sigmoid'),
          tf.keras.layers.Dense(128, activation='sigmoid'),
          tf.keras.layers.Dense(10)
          ])

        model_segment_full[i]._name = "model_full_segment"+str(i)
    
        model_segment_full[i].compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )
      

    elif percent == 0.75:
      #model_segment_full items represent the entirity of the model
      model_segment_full = list(segment_range)
      
      for i in segment_range:
        print('building model_segment_full[', i, '] using data segment', i)
        model_segment_full[i] = tf.keras.models.Sequential([
          root_model,
          tf.keras.layers.Dense(10)
          ])

        model_segment_full[i]._name = "model_full_segment"+str(i)

        model_segment_full[i].compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

    else:
      #model_segment_full items represent the entirity of the model
      model_segment_full = list(segment_range)
      
      for i in segment_range:
        print('building model_segment_full[', i, '] using data segment', i)

        temp_model = tf.keras.models.clone_model(basic_mnist_benchmark_model)
        layer_count = ceil(len(temp_model.layers)*(percent))
        print('the second half layer count is : ', len(temp_model.layers) - layer_count)
        # layer_count = len(temp_model.layers) - layer_count
        part_model = tf.keras.models.Sequential(temp_model.layers[layer_count:])

        model_segment_full[i] = tf.keras.models.Sequential([
          root_model,
          part_model
          ])

        # model_segment_full[i].summary()
        model_segment_full[i]._name = "model_full_segment"+str(i)

        model_segment_full[i].compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

  elif model_name == 'vgg':
    #model_segment_full items represent the entirity of the model
    model_segment_full = list(segment_range)
    
    for i in segment_range:
      print('building model_segment_full[', i, '] using data segment', i)

      temp_model = tf.keras.models.clone_model(vgg_benchmark_model)
      layer_count = ceil(len(temp_model.layers)*(percent))
      print('the second half layer count is : ', len(temp_model.layers) - layer_count)
      # layer_count = len(temp_model.layers) - layer_count
      part_model = tf.keras.models.Sequential(temp_model.layers[layer_count:])

      model_segment_full[i] = tf.keras.models.Sequential([
        root_model,
        part_model
        ])

      # model_segment_full[i].summary()
      model_segment_full[i]._name = "model_full_segment"+str(i)

      model_segment_full[i].compile(
          optimizer=tf.keras.optimizers.Adam(0.001),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
      )

  elif model_name == 'resnet':
    #model_segment_full items represent the entirity of the model
    model_segment_full = list(segment_range)
    
    for i in segment_range:
      print('building model_segment_full[', i, '] using data segment', i)

      temp_model = tf.keras.models.clone_model(resnet_benchmark_model)
      layer_count = ceil(len(temp_model.layers)*(percent))
      print('the second half layer count is : ', len(temp_model.layers) - layer_count)
      # layer_count = len(temp_model.layers) - layer_count
      part_model = tf.keras.models.Sequential(temp_model.layers[layer_count:])

      model_segment_full[i] = tf.keras.models.Sequential([
        root_model,
        part_model
        ])

      # model_segment_full[i].summary()
      model_segment_full[i]._name = "model_full_segment"+str(i)

      model_segment_full[i].compile(
          optimizer=tf.keras.optimizers.Adam(0.001),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
      )

  
  else:
    print('there was a problem building the root model')
    exit()

  return model_segment_full

def train_segments(segment_range, models, ds_train, ds_test):
  histories = list()
  c = 0
  latency_our_retraining = 0
  # training and retreiving predictions
  for i in segment_range:
    c += 1
    print('training models[', i, '] using data segment', i)
    
    start_our_retraining = t()

    histories.append(models[i].fit(
        ds_train[i],
        epochs=20,
        validation_data=ds_test,
        verbose = 1
    ))
    latency_our_retraining += (t() - start_our_retraining)

  latency_our_retraining /= c
  return models, histories, latency_our_retraining

def getDsTestDsInfo(ds_name):
  return tfds.load(
            ds_name, 
            split='test',
            as_supervised=True,
            with_info=True
        )
  
def get_latency_acc_dict(model_name, ds_name, seg_count, experiment_code, percent):
      accuracy_latency_dict = dict() #dictionary for collecting accuraccy and loss of different segment models

      # tf.keras.backend.clear_session()
      if experiment_code == "segment-count":
        mainKey = str(seg_count)
      elif experiment_code == "root-size":
        mainKey = str(percent)
      
      segment_range = range(seg_count)
        
      accuracy_latency_dict[mainKey] = dict()

      ds_train = load_data(seg_count, segment_range, ds_name)
      ds_train = segment_train_pipeline(segment_range, ds_train) #pipeline for the training data segments
      
      ds_test, ds_info = getDsTestDsInfo(ds_name)
      print(type(ds_test))
      ds_test = test_pipeline(ds_test) #pipeline for the test data

      if (model_name in ['vgg', 'resnet'] and ds_name in ['cifar10']) or (model_name in ['mnist-basic'] and ds_name in ['mnist']):
        print(f"Model: {model_name}\t\tDataset: {ds_name}" )
        root_model = build_root_iverted(model_name, percent)
        if root_model == None:
          print(f'there is nothing to run here when model name is {model_name} and percent is {percent}')
          return None, None, None

        model_segment_full = build_segment_models_inverted(model_name, segment_range, root_model, percent)

        # START latency of **training** our method
        start_our_training = t()
        # START latency of **retraining** our method
        start_our_retraining = t()

        model_segment_full, histories_before, latency_our_retraining = train_segments(segment_range, model_segment_full, ds_train, ds_test)

        # END latency of **training** our method
        latency_our_training = t() - start_our_training
        # END latency of **retraining** our method
        # latency_our_retraining = t() - start_our_retraining

        list_segment_acc = list()
        list_segment_loss = list()
        # START collecting all segment **prediction** latency
        latency_our_prediction = 0
        for i in segment_range:
            # START latency of **prediction**
            start_our_prediction = t()
            loss, acc = model_segment_full[i].evaluate(ds_test)
            # END latency of **prediction** 

            end_our_prediction = t()
            x=0
            for image, label in tfds.as_numpy(ds_test):
              x+=1

            latency_our_prediction += (end_our_prediction - start_our_prediction)/x
            list_segment_acc.append(acc)
            list_segment_loss.append(loss)

        # END collecting all segment **prediction** latency and divide by (/segment_count)
        latency_our_prediction = latency_our_prediction
        # latency_our_retraining = latency_our_retraining/segment_count

        accuracy_latency_dict[mainKey]['max_acc'] = float(round(max(list_segment_acc), 3))
        accuracy_latency_dict[mainKey]['min_acc'] = float(round(min(list_segment_acc), 3))
        accuracy_latency_dict[mainKey]['avg_acc'] = float(round(avg(list_segment_acc), 3))
        
        accuracy_latency_dict[mainKey]['max_loss'] = float(round(max(list_segment_loss), 3))
        accuracy_latency_dict[mainKey]['min_loss'] = float(round(min(list_segment_loss), 3))
        accuracy_latency_dict[mainKey]['avg_loss'] = float(round(avg(list_segment_loss), 3))

        accuracy_latency_dict[mainKey]['training_latency'] = float(latency_our_training)
        accuracy_latency_dict[mainKey]['retraining_latency'] = float(latency_our_retraining)
        accuracy_latency_dict[mainKey]['prediction_latency'] = float(latency_our_prediction)

        accuracy_latency_dict[mainKey]['root_percent'] = percent
        accuracy_latency_dict[mainKey]['dataset_name'] = ds_name

        print(accuracy_latency_dict[mainKey])

      else:
        print(f'there is nothing to run here when model name is {model_name} and dataset is {ds_name}')
        return None, None, None

      return accuracy_latency_dict

def build_root_HUSHDP(model_name, percent):
  if model_name == 'mnist-basic':
    baseline_dp_model = tf.keras.models.load_model('/Users/samaagazzaz/Desktop/UCSC_Courses/IPPML/experiments/source/hush_dp/mnist-basic/model.h5', compile=False)
  elif model_name == 'vgg':
    pass
  elif model_name == 'resnet':
    pass
  else:
    print('there was a problem building the root model, model name is not recognized')
    exit()

  layer_count = ceil(len(baseline_dp_model.layers)*percent)
  
  if layer_count < 1  or layer_count == len(resnet_benchmark_model.layers): 
    print('there was a problem building the root model')
    return None
  root_model = tf.keras.models.Sequential(baseline_dp_model.layers[:layer_count])
  for layer in root_model.layers:
      layer.trainable = False

  return root_model 

def build_segment_HUSHDP_models(model_name, segment_range, root_model, percent):
  HUSH_models = list(segment_range)
  for i in segment_range:
      print('building HUSH_models[', i, '] using data segment', i)
      if model_name == 'mnist-basic':
        temp_model = tf.keras.models.load_model('/Users/samaagazzaz/Desktop/UCSC_Courses/IPPML/experiments/source/hush_dp/mnist-basic/model.h5', compile=False)
      elif model_name == 'vgg':
        pass
      elif model_name == 'resnet':
        pass
      layer_count = ceil(len(temp_model.layers)*(percent))
      print('the second half layer count is : ', len(temp_model.layers) - layer_count)
      part_model = tf.keras.models.Sequential(temp_model.layers[layer_count:])

      HUSH_models[i] = tf.keras.models.Sequential([
          root_model,
          part_model
          ])

      HUSH_models[i].summary()
      HUSH_models[i]._name = "model_full_segment"+str(i)

      HUSH_models[i].compile(
          optimizer=tf.keras.optimizers.Adam(0.001),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
      )
  return HUSH_models

def build_root_HUSHDP_reveresed(model_name, percent):
  if model_name == 'mnist-basic':
    baseline_dp_model = tf.keras.models.load_model('/Users/samaagazzaz/Desktop/UCSC_Courses/IPPML/experiments/source/hush_dp/mnist-basic/model.h5', compile=False)
  elif model_name == 'vgg':
    pass
  elif model_name == 'resnet':
    pass
  else:
    print('there was a problem building the root model, model name is not recognized')
    exit()

  layer_count = ceil(len(baseline_dp_model.layers)*percent)
  
  if layer_count < 1  or layer_count == len(resnet_benchmark_model.layers): 
    print('there was a problem building the root model')
    return None
  root_model = tf.keras.models.Sequential(baseline_dp_model.layers[:layer_count])

  return root_model 

def build_segment_HUSHDP_models_reveresed(model_name, segment_range, root_model, percent):
  HUSH_models = list(segment_range)
  for i in segment_range:
      print('building HUSH_models[', i, '] using data segment', i)
      if model_name == 'mnist-basic':
        temp_model = tf.keras.models.load_model('/Users/samaagazzaz/Desktop/UCSC_Courses/IPPML/experiments/source/hush_dp/mnist-basic/model.h5', compile=False)
      elif model_name == 'vgg':
        pass
      elif model_name == 'resnet':
        pass
      layer_count = ceil(len(temp_model.layers)*(percent))
      print('the second half layer count is : ', len(temp_model.layers) - layer_count)
      part_model = tf.keras.models.Sequential(temp_model.layers[layer_count:])
      for layer in part_model.layers:
        layer.trainable = False

      HUSH_models[i] = tf.keras.models.Sequential([
          root_model,
          part_model
          ])

      HUSH_models[i].summary()
      HUSH_models[i]._name = "model_full_segment"+str(i)

      HUSH_models[i].compile(
          optimizer=tf.keras.optimizers.Adam(0.001),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
      )
  return HUSH_models

def get_latency_acc_dict_HUSHD_reveresed(model_name, ds_name, seg_count, experiment_code, percent):
      accuracy_latency_dict = dict() #dictionary for collecting accuraccy and loss of different segment models

      # tf.keras.backend.clear_session()
      if experiment_code == "segment-count":
        mainKey = str(seg_count)
      elif experiment_code == "root-size":
        mainKey = str(percent)
      
      segment_range = range(seg_count)
        
      accuracy_latency_dict[mainKey] = dict()

      ds_train = load_data(seg_count, segment_range, ds_name)
      ds_train = segment_train_pipeline(segment_range, ds_train) #pipeline for the training data segments
      
      ds_test, ds_info = getDsTestDsInfo(ds_name)
      print(type(ds_test))
      ds_test = test_pipeline(ds_test) #pipeline for the test data
      flag = "new"
      if (model_name in ['vgg', 'resnet'] and ds_name in ['cifar10']) or (model_name in ['mnist-basic'] and ds_name in ['mnist']):
        if flag == "old":
          print(f"Model: {model_name}\t\tDataset: {ds_name}" )
          root_model = build_root_HUSHDP_reveresed(model_name, percent)
          if root_model == None:
            print(f'there is nothing to run here when model name is {model_name} and percent is {percent}')
            return None, None, None

          HUSH_dp_models = build_segment_HUSHDP_models_reveresed(model_name, segment_range, root_model, percent)
        elif flag == "new":
          model = load_HUSH_dp_Model(model_name)

          model.trainable = False
          for i in range(ceil(len(model.layers)*percent)):
            model.layers[i].trainable = True

          HUSH_dp_models = list(segment_range)
          for i in segment_range:
            HUSH_dp_models[i] = tf.keras.models.clone_model(model)
            HUSH_dp_models[i].compile(
                # optimizer=get_optimizer('DP'),
                optimizer= tf.keras.optimizers.Adam(0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
            )

          

        # START latency of **training** our method
        start_our_training = t()
        # START latency of **retraining** our method
        start_our_retraining = t()

        HUSH_dp_models, histories_before, latency_our_retraining = train_segments(segment_range, HUSH_dp_models, ds_train, ds_test)

        # END latency of **training** our method
        latency_our_training = t() - start_our_training
        # END latency of **retraining** our method
        # latency_our_retraining = t() - start_our_retraining

        list_segment_acc = list()
        list_segment_loss = list()
        # START collecting all segment **prediction** latency
        latency_our_prediction = 0
        for i in segment_range:
            # START latency of **prediction**
            start_our_prediction = t()
            loss, acc = HUSH_dp_models[i].evaluate(ds_test)
            # END latency of **prediction** 

            end_our_prediction = t()
            x=0
            for image, label in tfds.as_numpy(ds_test):
              x+=1

            latency_our_prediction += (end_our_prediction - start_our_prediction)/x
            list_segment_acc.append(acc)
            list_segment_loss.append(loss)

        # END collecting all segment **prediction** latency and divide by (/segment_count)
        latency_our_prediction = latency_our_prediction
        # latency_our_retraining = latency_our_retraining/segment_count

        accuracy_latency_dict[mainKey]['max_acc'] = float(round(max(list_segment_acc), 3))
        accuracy_latency_dict[mainKey]['min_acc'] = float(round(min(list_segment_acc), 3))
        accuracy_latency_dict[mainKey]['avg_acc'] = float(round(avg(list_segment_acc), 3))
        
        accuracy_latency_dict[mainKey]['max_loss'] = float(round(max(list_segment_loss), 3))
        accuracy_latency_dict[mainKey]['min_loss'] = float(round(min(list_segment_loss), 3))
        accuracy_latency_dict[mainKey]['avg_loss'] = float(round(avg(list_segment_loss), 3))

        accuracy_latency_dict[mainKey]['training_latency'] = float(latency_our_training)
        accuracy_latency_dict[mainKey]['retraining_latency'] = float(latency_our_retraining)
        accuracy_latency_dict[mainKey]['prediction_latency'] = float(latency_our_prediction)

        accuracy_latency_dict[mainKey]['root_percent'] = percent
        accuracy_latency_dict[mainKey]['dataset_name'] = ds_name

        print(accuracy_latency_dict[mainKey])

      else:
        print(f'there is nothing to run here when model name is {model_name} and dataset is {ds_name}')
        return None, None, None

      return accuracy_latency_dict

def get_latency_acc_dict_HUSHDP(model_name, ds_name, seg_count, experiment_code, percent):
      accuracy_latency_dict = dict() #dictionary for collecting accuraccy and loss of different segment models

      # tf.keras.backend.clear_session()
      if experiment_code == "segment-count":
        mainKey = str(seg_count)
      elif experiment_code == "root-size":
        mainKey = str(percent)
      
      segment_range = range(seg_count)
        
      accuracy_latency_dict[mainKey] = dict()

      ds_train = load_data(seg_count, segment_range, ds_name)
      ds_train = segment_train_pipeline(segment_range, ds_train) #pipeline for the training data segments
      
      ds_test, ds_info = getDsTestDsInfo(ds_name)
      print(type(ds_test))
      ds_test = test_pipeline(ds_test) #pipeline for the test data
      flag = "new"
      if (model_name in ['vgg', 'resnet'] and ds_name in ['cifar10']) or (model_name in ['mnist-basic'] and ds_name in ['mnist']):
        if flag == "old":
          print(f"Model: {model_name}\t\tDataset: {ds_name}" )
          root_model = build_root_HUSHDP(model_name, percent)
          if root_model == None:
            print(f'there is nothing to run here when model name is {model_name} and percent is {percent}')
            return None, None, None

          HUSH_dp_models = build_segment_HUSHDP_models(model_name, segment_range, root_model, percent)
        elif flag == "new":
          model = load_HUSH_dp_Model(model_name)

          for i in range(ceil(len(model.layers)*percent)):
            model.layers[i].trainable = False

          HUSH_dp_models = list(segment_range)
          for i in segment_range:
            HUSH_dp_models[i] = tf.keras.models.clone_model(model)
            HUSH_dp_models[i].compile(
                # optimizer=get_optimizer('DP'),
                optimizer= tf.keras.optimizers.Adam(0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
            )

          

        # START latency of **training** our method
        start_our_training = t()
        # START latency of **retraining** our method
        start_our_retraining = t()

        HUSH_dp_models, histories_before, latency_our_retraining = train_segments(segment_range, HUSH_dp_models, ds_train, ds_test)

        # END latency of **training** our method
        latency_our_training = t() - start_our_training
        # END latency of **retraining** our method
        # latency_our_retraining = t() - start_our_retraining

        list_segment_acc = list()
        list_segment_loss = list()
        # START collecting all segment **prediction** latency
        latency_our_prediction = 0
        for i in segment_range:
            # START latency of **prediction**
            start_our_prediction = t()
            loss, acc = HUSH_dp_models[i].evaluate(ds_test)
            # END latency of **prediction** 

            end_our_prediction = t()
            x=0
            for image, label in tfds.as_numpy(ds_test):
              x+=1

            latency_our_prediction += (end_our_prediction - start_our_prediction)/x
            list_segment_acc.append(acc)
            list_segment_loss.append(loss)

        # END collecting all segment **prediction** latency and divide by (/segment_count)
        latency_our_prediction = latency_our_prediction
        # latency_our_retraining = latency_our_retraining/segment_count

        accuracy_latency_dict[mainKey]['max_acc'] = float(round(max(list_segment_acc), 3))
        accuracy_latency_dict[mainKey]['min_acc'] = float(round(min(list_segment_acc), 3))
        accuracy_latency_dict[mainKey]['avg_acc'] = float(round(avg(list_segment_acc), 3))
        
        accuracy_latency_dict[mainKey]['max_loss'] = float(round(max(list_segment_loss), 3))
        accuracy_latency_dict[mainKey]['min_loss'] = float(round(min(list_segment_loss), 3))
        accuracy_latency_dict[mainKey]['avg_loss'] = float(round(avg(list_segment_loss), 3))

        accuracy_latency_dict[mainKey]['training_latency'] = float(latency_our_training)
        accuracy_latency_dict[mainKey]['retraining_latency'] = float(latency_our_retraining)
        accuracy_latency_dict[mainKey]['prediction_latency'] = float(latency_our_prediction)

        accuracy_latency_dict[mainKey]['root_percent'] = percent
        accuracy_latency_dict[mainKey]['dataset_name'] = ds_name

        print(accuracy_latency_dict[mainKey])

      else:
        print(f'there is nothing to run here when model name is {model_name} and dataset is {ds_name}')
        return None, None, None

      return accuracy_latency_dict

def load_HUSH_dp_Model(model_name):
  model = tf.keras.models.load_model(f'/Users/samaagazzaz/Desktop/UCSC_Courses/IPPML/experiments/source/hush_dp/{model_name}/model.h5', compile=False)

  return model

def build_SISA_models(model_name, shard_range):
  if model_name == 'mnist-basic':
    #model_segment_full items represent the entirity of the model
    SISA_constituent_models = list(shard_range)
    
    for i in shard_range:
      print('building SISA_constituent_models[', i, '] using data segment', i)
      SISA_constituent_models[i] = tf.keras.models.clone_model(basic_mnist_benchmark_model)
      SISA_constituent_models[i]._name = "SISA_model_"+str(i)
      SISA_constituent_models[i].compile(
          optimizer=tf.keras.optimizers.Adam(0.001),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
      )
  elif model_name == 'vgg':
    #model_segment_full items represent the entirity of the model
    SISA_constituent_models = list(shard_range)
    
    for i in shard_range:
      print('building SISA_constituent_models[', i, '] using data segment', i)
      SISA_constituent_models[i] = tf.keras.models.clone_model(vgg_benchmark_model)
      SISA_constituent_models[i]._name = "SISA_model_"+str(i)
      SISA_constituent_models[i].compile(
          optimizer=tf.keras.optimizers.Adam(0.001),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
      )
  
  elif model_name == 'resnet':
    #model_segment_full items represent the entirity of the model
    SISA_constituent_models = list(shard_range)
    
    for i in shard_range:
      print('building SISA_constituent_models[', i, '] using data segment', i)
      SISA_constituent_models[i] = tf.keras.models.clone_model(resnet_benchmark_model)
      SISA_constituent_models[i]._name = "SISA_model_"+str(i)
      SISA_constituent_models[i].compile(
          optimizer=tf.keras.optimizers.Adam(0.001),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
      )
  
  
  else:
    print('there was a problem building the root model')
    exit()

  return SISA_constituent_models

def get_latency_acc_dict_SISA(model_name, ds_name, seg_count, experiment_code, percent = 0.5):
      accuracy_latency_dict = dict() #dictionary for collecting accuraccy and loss of different segment models

      if experiment_code == "segment-count":
        mainKey = str(seg_count)
      elif experiment_code == "root-size":
        mainKey = str(percent)
      
      shard_range = range(seg_count)
        
      accuracy_latency_dict[mainKey] = dict()

      ds_train = load_data(seg_count, shard_range, ds_name)
      ds_train = segment_train_pipeline(shard_range, ds_train) #pipeline for the training data segments
      
      ds_test, ds_info = getDsTestDsInfo(ds_name)
      ds_test = test_pipeline(ds_test) #pipeline for the test data

      if (model_name in ['vgg', 'resnet'] and ds_name in ['cifar10']) or (model_name in ['mnist-basic'] and ds_name in ['mnist']):
        print(f"Model: {model_name}\t\tDataset: {ds_name}" )

        SISA_models = build_SISA_models(model_name, shard_range)

        # START latency of **training** our method
        start_our_training = t()
        # START latency of **retraining** our method
        start_our_retraining = t()

        SISA_models, histories_before,latency_our_retraining = train_segments(shard_range, SISA_models, ds_train, ds_test)

        # END latency of **training** our method
        latency_our_training = t() - start_our_training
        # END latency of **retraining** our method
        # latency_our_retraining = t() - start_our_retraining

        list_segment_acc = list()
        list_segment_loss = list()
        # START collecting all segment **prediction** latency
        latency_our_prediction = 0
        for i in shard_range:
            # START latency of **prediction**
            start_our_prediction = t()
            loss, acc = SISA_models[i].evaluate(ds_test)
            # END latency of **prediction** 

            end_our_prediction = t()
            x=0
            for image, label in tfds.as_numpy(ds_test):
              x+=1

            latency_our_prediction += (end_our_prediction - start_our_prediction)/x
            list_segment_acc.append(acc)
            list_segment_loss.append(loss)

        # END collecting all segment **prediction** latency and divide by (/shard_count)
        # latency_our_retraining = latency_our_retraining/shard_count

        accuracy_latency_dict[mainKey]['max_acc'] = float(round(max(list_segment_acc), 3))
        accuracy_latency_dict[mainKey]['min_acc'] = float(round(min(list_segment_acc), 3))
        accuracy_latency_dict[mainKey]['avg_acc'] = float(round(avg(list_segment_acc), 3))
        
        accuracy_latency_dict[mainKey]['max_loss'] = float(round(max(list_segment_loss), 3))
        accuracy_latency_dict[mainKey]['min_loss'] = float(round(min(list_segment_loss), 3))
        accuracy_latency_dict[mainKey]['avg_loss'] = float(round(avg(list_segment_loss), 3))

        accuracy_latency_dict[mainKey]['training_latency'] = float(latency_our_training)
        accuracy_latency_dict[mainKey]['retraining_latency'] = float(latency_our_retraining)
        accuracy_latency_dict[mainKey]['prediction_latency'] = float(latency_our_prediction)

        accuracy_latency_dict[mainKey]['root_percent'] = percent
        accuracy_latency_dict[mainKey]['dataset_name'] = ds_name

        print(accuracy_latency_dict[mainKey])

      else:
        print(f'there is nothing to run here when model name is {model_name} and dataset is {ds_name}')
        return None, None, None

      return accuracy_latency_dict
 
def get_latency_acc_dict_benchmark(model_name, ds_name, differential_privacy):
  benchmark_model, benchmark_ds_train = build_benchmark(model_name, ds_name, differential_privacy)

  # START latency of **training** & **retraining** benchmark method
  start_benchmark_training = t()
  start_benchmark_retraining = t()
  benchmark_model = train_benchmark(benchmark_model, benchmark_ds_train)
  # END latency of **training** & **retraining** benchmark method
  latency_benchmark_training = t() - start_benchmark_training
  latency_benchmark_retraining = t() - start_benchmark_retraining


  ds_test, ds_info = getDsTestDsInfo(ds_name)
  ds_test = test_pipeline(ds_test) #pipeline for the test data

  # START latency of **prediction**
  start_benchmark_prediction = t()
  loss, acc = benchmark_model.evaluate(ds_test)
  # END latency of **prediction** 
  end_benchmark_prediction = t()

  x=0
  for image, label in tfds.as_numpy(ds_test):
    # print(type(image), type(label), label)
    x+=1
  latency_benchmark_prediction = (end_benchmark_prediction - start_benchmark_prediction)/x

  result =  {
            "max_acc": float(round(acc, 3)),
            "min_acc": float(round(acc, 3)),
            "avg_acc": float(round(acc, 3)),
            "training_latency": float(latency_benchmark_training),
            "retraining_latency": float(latency_benchmark_retraining),
            "prediction_latency": float(latency_benchmark_prediction),
            "dataset_name": ds_name
        
  }
  return result

def build_benchmark(model_name, ds_name, ds_info, withDP):

  if model_name == 'mnist-basic' and ds_name == 'mnist':
    learning_rate = 0.25 #, 'Learning rate for training')
    noise_multiplier = 0.1 #'Ratio of the standard deviation to the clipping norm')
    l2_norm_clip = 1.5 #, 'Clipping norm')
    microbatches = batch_size #, 'Number of microbatches '(must evenly divide batch_size)')

    
    opt = tf.keras.optimizers.Adam(0.001)
    los = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)
    if withDP:
      opt = DPKerasAdamOptimizer(l2_norm_clip= l2_norm_clip, noise_multiplier= noise_multiplier, num_microbatches= microbatches, learning_rate= learning_rate)
      los = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)

    benchmark_ds_train = tfds.load(
        ds_name, 
        split = 'train',
        as_supervised=True
      )
    benchmark_ds_train = benchmark_ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    benchmark_ds_train = benchmark_ds_train.cache()
    benchmark_ds_train = benchmark_ds_train.shuffle(ds_info.splits['train'].num_examples)
    benchmark_ds_train = benchmark_ds_train.batch(batch_size)
    benchmark_ds_train = benchmark_ds_train.prefetch(tf.data.AUTOTUNE)

    benchmark_model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='sigmoid'),
        tf.keras.layers.Dense(200, activation='sigmoid'),
        tf.keras.layers.Dense(128, activation='sigmoid'),
        tf.keras.layers.Dense(10)
      ])


    benchmark_model.compile(
        optimizer = opt,
        # Compute vector of per-example loss rather than its mean over a minibatch.
        loss = los,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
      )


  elif model_name == 'vgg' and ds_name == 'cifar10':
    learning_rate = 0.25 #, 'Learning rate for training')
    noise_multiplier = 0.1 #'Ratio of the standard deviation to the clipping norm')
    l2_norm_clip = 1.5 #, 'Clipping norm')
    microbatches = batch_size #, 'Number of microbatches '(must evenly divide batch_size)')

    
    opt = tf.keras.optimizers.Adam(0.001)
    los = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.losses.Reduction.NONE)
    if withDP:
      opt = DPKerasAdamOptimizer(l2_norm_clip= l2_norm_clip, noise_multiplier= noise_multiplier, num_microbatches= microbatches, learning_rate= learning_rate)
      los = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.losses.Reduction.NONE)

    benchmark_ds_train = tfds.load(
        ds_name, 
        split = 'train',
        as_supervised=True
      )
    benchmark_ds_train = benchmark_ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    benchmark_ds_train = benchmark_ds_train.cache()
    benchmark_ds_train = benchmark_ds_train.shuffle(ds_info.splits['train'].num_examples)
    benchmark_ds_train = benchmark_ds_train.batch(batch_size)
    benchmark_ds_train = benchmark_ds_train.prefetch(tf.data.AUTOTUNE)

    benchmark_model = tf.keras.models.clone_model(vgg_benchmark_model)

    benchmark_model.compile(
        optimizer = opt,
        # Compute vector of per-example loss rather than its mean over a minibatch.
        loss = los,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
      )

  elif model_name == 'resnet' and ds_name == 'cifar10':
    learning_rate = 0.25 #, 'Learning rate for training')
    noise_multiplier = 0.1 #'Ratio of the standard deviation to the clipping norm')
    l2_norm_clip = 1.5 #, 'Clipping norm')
    microbatches = batch_size #, 'Number of microbatches '(must evenly divide batch_size)')

    
    opt = tf.keras.optimizers.Adam(0.001)
    los = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.losses.Reduction.NONE)
    if withDP:
      opt = DPKerasAdamOptimizer(l2_norm_clip= l2_norm_clip, noise_multiplier= noise_multiplier, num_microbatches= microbatches, learning_rate= learning_rate)
      los = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.losses.Reduction.NONE)

    benchmark_ds_train = tfds.load(
        ds_name, 
        split = 'train',
        as_supervised=True
      )
    benchmark_ds_train = benchmark_ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    benchmark_ds_train = benchmark_ds_train.cache()
    benchmark_ds_train = benchmark_ds_train.shuffle(ds_info.splits['train'].num_examples)
    benchmark_ds_train = benchmark_ds_train.batch(batch_size)
    benchmark_ds_train = benchmark_ds_train.prefetch(tf.data.AUTOTUNE)

    benchmark_model = tf.keras.models.clone_model(resnet_benchmark_model)

    benchmark_model.compile(
        optimizer = opt,
        # Compute vector of per-example loss rather than its mean over a minibatch.
        loss = los,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
      )


  else:
    print(f'there is no match benchmark case for model name: {model_name} and dataset {ds_name}')
    return None, None




  benchmark_model._name = "benchmark_model"
  if withDP:
    benchmark_model._name = "benchmark_withDP_model"

  return benchmark_model, benchmark_ds_train
  
def train_benchmark(benchmark_model, benchmark_ds_train):
  benchmark_model.fit(
          benchmark_ds_train,
          epochs=20,
          verbose = 0
      )
  dt = get_datatime()
  benchmark_model.save_weights('hush_dp/mnist/weights_'+dt)
  return benchmark_model

def get_datatime():
  from datetime import datetime
  now = datetime.now()
  # dd/mm/YY H:M:S
  dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
  return dt_string

def printdict(d):
  print(json.dumps(str(d), indent=4)) # or d.item() instead of str(d)

def saveresultsfile(argumnets, model, results, approach):
  result_json_name = f"{argumnets[1]}-{argumnets[2]}-{model}-{approach}.json"
  path = os.path.split(os.getcwd())[0]
  results_Path = path + f"/experiments/results/{argumnets[1]}/{model}/"
  print(results_Path)
  print(results_Path + result_json_name)

  if os.path.exists(results_Path + result_json_name):
    os.remove(results_Path + result_json_name)

  if not os.path.exists(results_Path):
    os.makedirs(results_Path)
    
  with open(results_Path + result_json_name, "w") as result_file:
    json.dump(results, result_file, indent=2)

def merge(dict1, dict2):
    dict1.update(dict2)
