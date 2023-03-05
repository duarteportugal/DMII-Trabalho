#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


# In[25]:


mnist = keras.datasets.mnist
(X_train_full, y_train_full), (X_test,y_test) = mnist.load_data()  


# In[26]:


X_train_full.shape


# In[27]:


X_test.shape


# In[28]:


X_train_full[0]


# In[29]:


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10,10))
a = 0

for i in range(3):
    for j in range(3):
        axes[i,j].imshow(X_train_full[a], cmap=plt.get_cmap('gray'))
        a = a + 1
        
plt.show()


# In[30]:


X_valid, X_train = X_train_full[:5000]/255, X_train_full[5000:]/255
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test/255


# In[31]:


class_names = ['0','1','2','3','4','5','6','7','8','9']


# In[34]:


class_names[y_train[3]]


# In[35]:


plt.imshow(X_train[3],cmap=plt.get_cmap('gray'))


# In[42]:


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))


# In[43]:


model.summary()


# In[44]:


model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


# In[48]:


history = model.fit(X_train, y_train, epochs=30,validation_data = (X_valid, y_valid), batch_size=32)


# In[49]:


import pandas as pd

pd.DataFrame(history.history).plot(figsize=(15,8))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()


# In[50]:


model.evaluate(X_test,y_test)


# In[58]:


y_prob = model.predict(X_test)
y_classes = y_prob.argmax(axis=1)
confusion_matrix = tf.math.confusion_matrix(y_test, y_classes)


# In[59]:


import seaborn as sb

fig = sb.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues')

fig.set_xlabel('Predicted labels')
fig.set_ylabel('True labels')
fig.set_title('Confusion Matrix')
fig.xaxis.set_ticklabels(class_names)
fig.yaxis.set_ticklabels(class_names)
fig.figure.set_size_inches(10,10)

plt.show()


# In[ ]:




