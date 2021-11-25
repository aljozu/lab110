#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
import face_recognition


# In[29]:


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


# In[14]:


from random import randint


# In[15]:


p1 = face_recognition.face_distance(puntos, puntos[randint(0,50)])
p2 = face_recognition.face_distance(puntos, puntos[randint(0,50)])


# In[16]:


import matplotlib.pyplot as plt
import numpy as np


# In[17]:


f1=list(np.append(p1,p2))
plt.hist(f1)
plt.show()


# In[22]:


from rtree import index


# In[32]:


p = index.Property()
p.dimension = 128
p.buffering_capacity = 16
p.dat_extension = 'data'
p.idx_extension = 'index'
idx = index.Index('128d_index', properties=p)


# In[33]:


for i in range(0, len(puntos)):
    idx.insert(i, tuple(puntos[i])*2)


# In[35]:


#Range Search
def range_search(img, r):
    picture = face_recognition.load_image_file(img)
    known_face_encoding = face_recognition.face_encodings(picture)[0]
    ans = []
    for i in puntos:
        if len(face_recognition.face_encodings(i))>0:
            image_compare_encoding = face_recognition.face_encodings(i)[0]
            dist = face_recognition.face_distance(known_face_encoding, image_compare_encoding)
            if dist < r:
                result.append(i)
    return result


# In[ ]:


#KNN Search
def KNN_search(img, k):
    picture = face_recognition.load_image_file(img)
    known_face_encoding = face_recognition.face_encodings(picture)[0]
    ans = PriorityQueue()
    for i in puntos:
        if len(face_recognition.face_encodings(i))>0:
            image_compare_encoding = face_recognition.face_encodings(i)[0]
            dist = face_recognition.face_distance(known_face_encoding, image_compare_encoding)
            ans.put((dist, i))
    final_ans = []
    for i in range(k):
        final_ans.append(result.get())
    return final_ans

