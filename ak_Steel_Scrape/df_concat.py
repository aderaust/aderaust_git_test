#!/usr/bin/env python
# coding: utf-8

# In[1]:


from AK_Scrape_FUNC import *


# In[2]:


# scraping ak for new data

current_df = data_getter()


# In[3]:



current_df_clean = current_df.dropna()


# In[5]:


# reading in old df 

old_df = pd.read_csv('/Users/austinader/ak_Steel_Scrape/clean_ak_df', dtype = {'defect_code': 'object'}, float_precision='round_trip')


# In[7]:


# concat old
new_df = pd.concat([old_df, current_df], ignore_index=True)


# In[8]:



final_new_df = new_df.drop_duplicates()


# In[14]:



final_new_df.to_csv('/Users/austinader/ak_Steel_Scrape/clean_ak_df', index=False)

