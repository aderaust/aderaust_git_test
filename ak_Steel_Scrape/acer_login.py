#!/usr/bin/env python
# coding: utf-8

# In[1]:


from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.chrome.options import Options

import os
import time
import numpy as np

from datetime import date


# ## creating current directory everyday

# In[2]:


desired_directory = '/Users/austinader/ak_Steel_Scrape/raw_acermital_dfs'



def create_directory():

    os.chdir(desired_directory)
    current_day = date.today()

    current_day_pretty = current_day.strftime("%m_%d_%y")



    if current_day_pretty not in os.listdir():  ## make directory named today's date if todays directory is not already in working directory
        os.makedirs(current_day_pretty)
       
    final_directory = desired_directory + '/' + current_day_pretty
    
    os.chdir(final_directory)
    
    return final_directory
        


# ## setting chrome options and opening the webpage

# In[3]:


todays_directory = create_directory()

# chromeOptions = Options()

# chromeOptions.add_experimental_option("prefs",{'download.default_directory': todays_directory})

# browser = webdriver.Chrome(ChromeDriverManager().install(), options= chromeOptions)
# browser.get('https://pc.arcelormittal.com/partnercentre/Login')
# test = browser.find_element_by_css_selector('button')




# In[4]:


## creating function to log in and pull every dataframe


def log_n_pull():
    chromeOptions = Options()

    chromeOptions.add_experimental_option("prefs",{'download.default_directory': todays_directory})

    browser = webdriver.Chrome(ChromeDriverManager().install(), options= chromeOptions)
    browser.get('https://pc.arcelormittal.com/partnercentre/Login')
    test = browser.find_element_by_css_selector('button')
    
    time.sleep(1 + np.random.random())

    email_cred = browser.find_element_by_id("defaultLoginFormEmail")


    password_cred = browser.find_element_by_id("defaultLoginFormPassword")

    time.sleep(1 + np.random.random())


    email_cred.send_keys('msimone')

    time.sleep(1 + np.random.random())

    password_cred.send_keys('targetsteel1')

    time.sleep(1 + np.random.random())

    browser.find_element_by_xpath('/html/body/main/div/div/div/div[3]/div[1]/div/div/div/form/div[6]/button').click()

    time.sleep(1 + np.random.random())


    browser.get('https://pc.arcelormittal.com/PartnerCentre/inv4Sale_InvConsReport.asp?RptID=53')

    time.sleep(1 + np.random.random())


    browser.find_element_by_xpath('//*[@id="butEXCELReport1"]').click()
    
    
    count_raw = browser.find_element_by_xpath("/html/body/table[3]/tbody/tr/td[2]/table/tbody/tr/td/form/table[1]/tbody/tr[5]/td[2]").text

    count_raw_list = [x for x in count_raw.split()]  

    count_list = [int(item) for subitem in count_raw_list for item in subitem.split() if item.isdigit()]

    page_count = max(count_list)


    for i in range(page_count - 1):
        time.sleep(np.random.rand())
        browser.find_element_by_xpath('//*[@id="spanPAGE1"]/a[2]').click()
    


# In[5]:


log_n_pull()

