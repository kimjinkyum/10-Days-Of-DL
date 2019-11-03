#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[35]:


#make constant

t = tf.constant([[[[9, 10, 11, 12]],
                   [[13, 14, 15, 16]]]])

t1=tf.constant([[[1, 2, 3, 4], [5, 6, 7, 8]],
              [[17, 18, 19, 20], [21, 22, 23, 24]]])


print(t.ndim)
print(t.shape)

print(t1.ndim)
print(t1.shape)


# In[29]:


#reduce sum
print(tf.reduce_sum(t1,axis=0))

print(tf.reduce_sum(t1,axis=1))

print(tf.reduce_sum(t1,axis=2))


# In[31]:


#broadcasting
m1=tf.constant([3,6])
m2=tf.constant([10])
print(m1+m2)


# In[42]:


#squeeze
print(t.shape)
print(tf.squeeze(t).shape)
print(tf.squeeze(t,axis=0).shape)
#print(tf.squeeze(t,axis=1).shape) ##ERROR
print(tf.squeeze(t,axis=2).shape)


# In[61]:


#expand
print(t.shape)
print(tf.expand_dims(t,[1]).shape)
print(tf.expand_dims(t,[3]).shape)
print(tf.expand_dims(t,[4]).shape)
#print(tf.expand_dims(t,[5]).shape) #Error!


# In[63]:


tf.cast(t1 , tf.float32)

