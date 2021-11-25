#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import face_recognition


# In[3]:


rootdir = 'C:\\Users\\lojaz\\Desktop\\script\\lfw'
puntos = []
cont = 0
for subdir, dirs, files in os.walk(rootdir):
    cont+=1
    for file in files:
        ruta = os.path.join(subdir, file)
        picture = face_recognition.load_image_file(ruta)
        known_face_encoding = face_recognition.face_encodings(picture)[0]
        #print(known_face_encoding)
        puntos.append(known_face_encoding)
    if cont == 50:
        break


# In[4]:


from random import randint


# In[5]:


p1 = face_recognition.face_distance(puntos, puntos[randint(0,50)])
p2 = face_recognition.face_distance(puntos, puntos[randint(0,50)])


# In[6]:


import matplotlib.pyplot as plt
import numpy as np


# In[7]:


f1=list(np.append(p1,p2))
plt.hist(f1)
plt.show()


# In[8]:


from rtree import index


# In[9]:


p = index.Property()
p.dimension = 128
p.buffering_capacity = 16
p.dat_extension = 'data'
p.idx_extension = 'index'
idx = index.Index('128d_index', properties=p)


# In[10]:


for i in range(0, len(puntos)):
    idx.insert(i, tuple(puntos[i])*2)

