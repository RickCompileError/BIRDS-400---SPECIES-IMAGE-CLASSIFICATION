#!/usr/bin/env python
# coding: utf-8

# # Import 函式庫及設備測試

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[2]:


#used to check the current environment
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

tf.__version__


# # 讀取及建立 Dataset

# In[3]:


data_dir = "CUB_200_2011/images"
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)


# In[4]:


roses = list(data_dir.glob('001.Black_footed_Albatross/*'))
PIL.Image.open(str(roses[3]))


# In[5]:


batch_size = 32
img_height = 224
img_width = 224


# In[6]:


datagen = ImageDataGenerator(
  rotation_range=40,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  fill_mode='nearest')

#datagen.fit(train_ds)


# In[23]:


train_ds = datagen.flow_from_directory(
  data_dir,
  class_mode='categorical',
  seed=123,
  target_size=(img_height, img_width),
  batch_size=batch_size)


# In[24]:


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  label_mode='categorical',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# In[11]:


print(train_ds)


# In[13]:


class_names = train_ds.class_indices
print(class_names)


# In[14]:


for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


# In[15]:


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# In[16]:


normalization_layer = layers.Rescaling(1./255)


# In[17]:


num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        
  layers.Conv2D(16, 5, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.25),
    
  layers.Conv2D(32, 5, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.25),
    
  layers.Conv2D(64, 5, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.25),
    
  layers.Conv2D(128, 5, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.25),
    
  layers.Flatten(),  
  
  layers.Dense(512, activation='relu', kernel_initializer='normal'),
  layers.Dropout(0.25),
  layers.Dense(num_classes, activation='softmax', kernel_initializer='normal')
])


# In[18]:


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[19]:


model.summary()


# In[ ]:


epochs=10

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
)


# In[22]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)
#epochs_range = range(11)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[23]:


#data augmentation
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal_and_vertical", input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(0.5),
    layers.RandomZoom(0.5),
  ]
)


# In[24]:


plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")


# In[25]:


model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.25),

  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.25),
    
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.25),
    
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.25),
        
  layers.Flatten(),  
  
  layers.Dense(512, activation='relu', kernel_initializer='normal'),
  layers.Dropout(0.25),
  layers.Dense(num_classes, activation='softmax', kernel_initializer='normal')
])


# In[26]:


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[27]:


model.summary()


# In[30]:


epochs=50

EARLY_STOPPING = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  #callbacks = [EARLY_STOPPING]
)


# In[29]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)
#epochs_range = range(11)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[ ]:




