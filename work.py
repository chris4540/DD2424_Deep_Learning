#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import matplotlib.pyplot as plt


# In[24]:


df = pd.read_json("a2_ex4_coarse_search.json")


# In[25]:


df = df.drop("scores", axis="columns")


# In[36]:


df.plot(x="lambda_", y="mean_score", kind="scatter", logx=True)


# In[39]:


df = df.sort_values(by=['lambda_'])


# In[40]:


df.plot(x="lambda_", y="mean_score", kind="line", logx=True)


# In[43]:


df2 = df.sort_values(by=['mean_score'], ascending=False)


# In[46]:


df2['log_lambda'] = np.log10(df2['lambda_'])


# In[47]:


df2


# In[ ]:




