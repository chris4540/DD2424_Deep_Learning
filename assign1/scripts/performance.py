#!/usr/bin/env python
# coding: utf-8

# This file shows the performance between f2py and pure numpy matmul performacne

# See also: https://www.benjaminjohnston.com.au/matmul
# In[1]:


import numpy as np
import lib_clsr
import lib_clsr.ann
import lib_clsr.ann_f


# In[2]:


ann_py = lib_clsr.ann
ann_f  = lib_clsr.ann_f.ann_for  # ann_f compiled with gfortran -O3 -fexternal-blas --link-openblas


# In[3]:


k = 12
d = 4000
n = 100
W_mat = np.random.randn(k ,d)
X_mat = np.random.randn(d ,n) * np.random.randint(2)
b_vec = np.random.randn(k ,1)
Y_mat = np.eye(k)[np.random.choice(k, n)].T
lambda_ = 0.01
s_mat = W_mat.dot(X_mat) + b_vec


# In[4]:


X_mat_f = np.asfortranarray(X_mat)
W_mat_f = np.asfortranarray(W_mat)
b_vec_f = np.asfortranarray(b_vec)
Y_mat_f = np.asfortranarray(Y_mat)


# f2py -c ann_f.pyf -I. --opt="-O3 -fexternal-blas" --link-openblas ann_f.f90

# In[5]:


get_ipython().run_line_magic('timeit', 'ann_py.softmax(s_mat)')


# In[6]:


get_ipython().run_line_magic('timeit', 'ann_f.softmax(s_mat)')


# In[7]:


get_ipython().run_line_magic('timeit', 'ann_py.evaluate_classifier(X_mat, W_mat, b_vec)')


# In[8]:


get_ipython().run_line_magic('timeit', 'ann_f.evaluate_classifier(X_mat_f, W_mat_f, b_vec_f)')


# In[9]:


get_ipython().run_line_magic('timeit', 'ann_py.compute_gradients(X_mat, Y_mat, W_mat, b_vec, lambda_)')


# In[10]:


get_ipython().run_line_magic('timeit', 'ann_f.compute_gradients(X_mat_f, Y_mat_f, W_mat_f, b_vec_f, lambda_)')


# In[ ]:




