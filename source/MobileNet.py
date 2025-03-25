# https://keras.io/api/applications/
# https://www.kaggle.com/code/sanjaybalamurugan/mobilenet-for-cifar-10-dataset
# https://stackoverflow.com/questions/52209851/expected-validation-accuracy-for-keras-mobile-net-v1-for-cifar-10-training-from
#%%
from keras.utils.vis_utils import plot_model
import tensorflow as tf
from utils import * 


# build an evaluation pipeline
def test_pipeline(ds_test):
  ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
  ds_test = ds_test.batch(128)
  ds_test = ds_test.cache()
  ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

  return ds_test


# Load all segments of the train split and create the input pipeline
def load_data(segment_count, segment_range, ds_name):
  train_splits = tfds.even_splits('train', n=segment_count)
  ds_train = list(segment_range)

  for i in segment_range:
    ds_train[i] = tfds.load(
      ds_name, 
      split=train_splits[i],
      as_supervised=True,
    )
    print('training segment ', i+1, 'size', len(ds_train[i]))
    
  return ds_train

  # build a training pipeline and pass each segment through it
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

def segment_train_pipeline(segment_range, ds_train):
  for i in segment_range:
    ds_train[i] = ds_train[i].map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train[i] = ds_train[i].cache()
    ds_train[i] = ds_train[i].shuffle(len(ds_train[i]))
    ds_train[i] = ds_train[i].batch(128)
    ds_train[i] = ds_train[i].prefetch(tf.data.AUTOTUNE)
  return ds_train



# TODO check to make sure that adding the model layer by layer removes the underlying information about the layer including whether it should be trainable or not
# TODO check why here the model has very high testing results but not in the main function
class MobileNet:
  def __init__(self):
    # load a copy of the mobilenet model
    MobileNet_model = tf.keras.applications.MobileNet(
        input_shape=None,
        alpha=1.0,
        depth_multiplier=1,
        dropout=0.001,
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
        classes=10,
        classifier_activation="softmax"
    )

    # MobileNet_model.trainable = False


    # print(len(MobileNet_model.layers))
    self.model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(32,32,3),name='input')
    ])
    # MobileNet_model.summary()
    # plot_model(MobileNet_model)
    # MobileNet_model.trainable = False 

    c = len(MobileNet_model.layers)/3
    for layer in MobileNet_model.layers:
        if c > 0:
            layer.trainable = False
            c-=1
        self.model.add(layer)
        # print(layer.trainable)


    self.model.add(tf.keras.layers.GlobalAveragePooling2D())
    self.model.add(tf.keras.layers.Dense(1024,activation=('relu')))
    self.model.add(tf.keras.layers.Dense(512,activation=('relu')))
    self.model.add(tf.keras.layers.Dense(256,activation=('relu')))
    self.model.add(tf.keras.layers.Dropout(0.5))
    self.model.add(tf.keras.layers.Dense(128,activation=('relu')))
    self.model.add(tf.keras.layers.Dropout(0.5))
    self.model.add(tf.keras.layers.Dense(10,activation=('softmax')))

    plot_model(self.model)


# ds_train = load_data(1, [0], 'cifar10')
# ds_train = segment_train_pipeline([0], ds_train) 

# ds_test, ds_info = tfds.load(
#     'cifar10', 
#     split='test',
#     shuffle_files=True,
#     as_supervised=True,
#     with_info=True
# )
# ds_test = test_pipeline(ds_test) 

# opt = tf.keras.optimizers.Adam(0.001)
# los = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# x = MobileNet()

# x.model.compile(
#         optimizer = opt,
#         # Compute vector of per-example loss rather than its mean over a minibatch.
#         loss = los,
#         metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
#       )

# x.model.fit(
#         ds_train[0],
#         epochs=6,
#         validation_data=ds_test,
#         verbose = 1
#     )


# #%%
# plot_model(x.model, show_shapes=True, show_layer_names=True)

# #%%
# loss, acc = x.model.evaluate(ds_test)
# acc


