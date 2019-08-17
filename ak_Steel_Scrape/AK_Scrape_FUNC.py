#!/usr/bin/env python
# coding: utf-8

# In[14]:


import requests
from bs4 import BeautifulSoup as soup
import numpy as np
import pandas as pd
import re
import string
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler  


from sklearn.neural_network import MLPRegressor


from sklearn.model_selection import train_test_split


# In[ ]:





# In[15]:


# username: E76723
# password: target01
# login-form-type: pwd


# In[16]:



def log_in():
    '''This function logs the scrapper into the main menu with the approiate cookies'''
    login_data = {'username': 'E76723',
                  'password': 'target01',
                 'login-form-type': 'pwd'}

    
    session = requests.Session()
    
    # initial login page 
    
    login_link = "https://www.aksteelonline.com/webapp/wcs/stores/servlet/StoreView?storeId=10001&langId=-1"
    
    login_screen = session.get(login_link)

#     login_soup = soup(login_screen.content)

    # this f value is dynamic

#     login_data['f'] = login_soup.find('input', {'name': 'f'})['value']

   
    
    # form request link
    form_url = 'https://www.aksteelonline.com/pkmslogin.form'

    # sending login data into form
    main_page = session.post(form_url, login_data)
    
    # retreiving catalog page
    catalog_page = session.get(
        'https://www.aksteelonline.com/webapp/wcs/stores/servlet/LogonForm?storeId=10001&catalogId=10001')
    
    return session


# In[17]:


session_logged_in = log_in()


# In[18]:


# returning and converting BUY_NOW auctions

buy_now_raw = session_logged_in.get('https://www.aksteelonline.com/webapp/wcs/stores/servlet/CatalogSearchResultView?storeId=10001&catalogId=10001&ButtonB=Search+Buy+Now&countPerPage=50&sortSelect=product&page=0').text

buy_now = soup(buy_now_raw, 'html.parser')


# In[19]:



# converting the table

buy_now_table = buy_now.find('table', {'border': "0", 'cellpadding':'2'})


# In[20]:


# creating list of prices

def price_getter(buy_now_table):

    price = buy_now_table.find_all('td', {'headers': 'WC_CatalogSearchResultDisplay_PriceHeader'})

    price_list = []
    for i in range(len(price)):

        price_list.append(float(re.sub(r'[^\d.]+', '', price[i].text)))
        
    return price_list


# In[21]:


# creating list of gauges


def gauge_getter(buy_now_table):
    gauge = buy_now_table.find_all('td', {'headers': 'WC_CatalogSearchResultDisplay_GAHeader'})


    gauge_list = []
    for i in range(len(gauge)):

        gauge_list.append(float(re.sub(r'[^\d.]+', '', gauge[i].text)))
        
    return gauge_list


# In[22]:


# creating list of widths

def width_getter(buy_now_table):

    width = buy_now_table.find_all('td', {'headers': 'WC_CatalogSearchResultDisplay_WDHeader'})


    width_list = []
    for i in range(len(width)):

        width_list.append(float(re.sub(r'[^\d.]+', '', width[i].text)))
        
    return width_list


# In[23]:


# creating list of weights

def weight_getter(buy_now_table):

    weight = buy_now_table.find_all('td', {'headers': 'WC_CatalogSearchResultDisplay_WeightHeader'})


    weight_list = []
    for i in range(len(weight)):

        weight_list.append(float(re.sub(r'[^\d.]+', '', weight[i].text)))
        
    return weight_list


# In[24]:


# creating list of locations


        
def location_getter(buy_now_table):

    location = buy_now_table.find_all('td', {'headers': 'WC_CatalogSearchResultDisplay_LocationHeader'})


    location_list = []
    for i in range(len(location)):

        location_list.append(location[i].text.strip('\r\n\t\t\t\t\t'))
    return location_list
    


# In[25]:


# scraping index table
index_table = buy_now_table.find_all('td', {'align':'left'})


# In[26]:


# creating index list of unique codes

# table cellpadding="0" cellspacing="0" border="0" width="100%"


def index_getter(index_table):
    index_list = []

    for i in range(len(index_table)):
        index_list.append(index_table[i].text.rstrip().strip('\r\n\t\t\t\t\t\t\t\t\t\xa0\xa0\xa0\r\n\t\t\t\t\t\t\t\t\t\r\n\t\t\t\t\t\t\t\t\t\t'))
    return index_list


# In[27]:


# scraping inner-table

inner_table = buy_now_table.find_all('table', {'cellpadding' : "0", 'cellspacing' : "0", 'border' : "3",
                                    'width' : "100%"})


# In[28]:


# creating list of defects

def defect_getter(inner_table):

    defect_list = []

    for i in range(len(inner_table)):

        if inner_table[i].find(text = re.compile(r"\bDefect\b")) is None:
            defect_list.append(None)
        else:
            defect_list.append(inner_table[i].find(text = re.compile(r"\bDefect\b")).findNext('td').text.rstrip().strip('\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t'))
    return defect_list
#     print(i)

# all_defects = buy_now_table.find_all(text = re.compile(r"\bDefect\b"))    

# buy_now_table.find(text = re.compile(r"\bDefect\b")).findNext('td').text



# In[29]:


# creating list of lin-feet

def linfeet_getter(inner_table):
    
    lin_feet_list = []
    for i in range(len(inner_table)):

        if inner_table[i].find(text = re.compile("Lin. Ft.")) is None:
            lin_feet_list.append(None)
        else:
            lin_feet_list.append(float(inner_table[i].find(text = re.compile("Lin. Ft.")).findNext('td').text.rstrip().strip('\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t').replace(",","")))
            
    return lin_feet_list
        
        


# In[31]:


# retreiving chemistry data

chem_table = buy_now_table.find_all(text = re.compile("C-"))

def chem_getter(chem_table):

    chem_list = []
    for i in range(len(chem_table)):


        digit_string = re.sub('[^0-9\.]',' ', chem_table[i].rstrip())
        
#         if digit_string is None:
#             chem_list.append(None)
            
#             pass
        

        pretty_string= [float(x) for x in digit_string.split()]

        #logic if code does or does not contain heat number

        
        
        
        if pretty_string[0] > 100:
            pretty_string.pop(0)
        
        if len(pretty_string) != 8:
            chem_list.append(np.repeat(np.nan, 8))
        

        
        else: chem_list.append(pretty_string)
            
            

    return np.array(chem_list)


chem_getter_test = chem_getter(chem_table)


# In[33]:


# chem_list = []
# pretty_string_test = []
# for i in range(len(chem_table)):


#     digit_string = re.sub('[^0-9\.]',' ', chem_table[i].rstrip())

#     if digit_string is None:
#         chem_list.append(None)

#         pass


#     pretty_string= [float(x) for x in digit_string.split()]
#     pretty_string_test.append(pretty_string)

# #     #logic if code does or does not contain heat number

# #     if pretty_string[0] > 100:
# #         pretty_string.pop(0)

# #     if pretty_string =! 
    
    
#     chem_list.append(pretty_string)



# digit_string.split()


# In[34]:


def url_generator(buy_now):
    url_count = int(buy_now.find('a', {'id' :"WC_PageURL_Link_Next"}).find_previous_sibling('a').text.rstrip().strip('\r\n\t\t\t\t\t'))

    skeleton_url = "https://www.aksteelonline.com/webapp/wcs/stores/servlet/CatalogSearchResultView?storeId=10001&catalogId=10001&ButtonB=Search+Buy+Now&countPerPage=50&sortSelect=product&page="


    page_urls = []
    for i in range(url_count):
        page_urls.append(skeleton_url + str(i))
    return page_urls


# In[35]:


def data_getter():

    price_master_list = []
    gauge_master_list = []
    width_master_list = []
    weight_master_list = []
    location_master_list = []
    defect_master_list = []
    linfeet_master_list = []
    index_master_list = []
    
    

    # chemistry lists 
    
    carbon_master = []
    manganese_master = []
    phosphorus_master = []
    sulfur_master = []
    silicon_master = []
    aluminium_master = []
    niobium_master = []
    vanadium_master = []

    
    # logging into session
    session_logged_in = log_in()
    
    # creating input for url generator
    buy_now_raw_url = session_logged_in.get('https://www.aksteelonline.com/webapp/wcs/stores/servlet/CatalogSearchResultView?storeId=10001&catalogId=10001&ButtonB=Search+Buy+Now&countPerPage=50&sortSelect=product&page=0').text

    buy_now_url = soup(buy_now_raw_url, 'html.parser')
    
    
    # generating urls for loop
    page_urls = url_generator(buy_now_url)
    

    for page in page_urls:
        
        # retreiving/updating all required variables for each iteration of loop

        buy_now_raw = session_logged_in.get(page).text

        buy_now = soup(buy_now_raw, 'html.parser')
        
        buy_now_table = buy_now.find('table', {'border': "0", 'cellpadding':'2'})
        
        index_table = buy_now_table.find_all('td', {'align':'left'})
        
        
        inner_table = buy_now_table.find_all('table', {'cellpadding' : "0", 'cellspacing' : "0", 'border' : "3",
                                        'width' : "100%"})
        
        chem_table = buy_now_table.find_all(text = re.compile("C-"))
        
    
        

        # price

        current_price = price_getter(buy_now_table)

        price_master_list.extend(current_price)


        # gauges

        current_gauge = gauge_getter(buy_now_table)

        gauge_master_list.extend(current_gauge)
        

        # widths

        current_width = width_getter(buy_now_table)

        width_master_list.extend(current_width)

        
        # weights

        current_weight = weight_getter(buy_now_table)

        weight_master_list.extend(current_weight)

        
        # locations

        current_location = location_getter(buy_now_table)

        location_master_list.extend(current_location)

        ###


        # defects

        current_defect = defect_getter(inner_table)

        defect_master_list.extend(current_defect)


        # linfeets

        current_linfeet = linfeet_getter(inner_table)

        linfeet_master_list.extend(current_linfeet) 


        ###

        # indexs

        current_index = index_getter(index_table)

        index_master_list.extend(current_index)


        ###


        # pulling all chemicals

        current_chem = chem_getter(chem_table)


        carbon_master.extend(current_chem[:,0].tolist())
        manganese_master.extend(current_chem[:,1].tolist())
        phosphorus_master.extend(current_chem[:,2].tolist())
        sulfur_master.extend(current_chem[:,3].tolist())
        silicon_master.extend(current_chem[:,4].tolist())
        aluminium_master.extend(current_chem[:,5].tolist())
        niobium_master.extend(current_chem[:,6].tolist())
        vanadium_master.extend(current_chem[:,7].tolist())


    comprehensive_dict = {"index" : index_master_list, 'price' : price_master_list ,'carbon' : carbon_master, 'manganese' : manganese_master,
    'phosphorus': phosphorus_master, 'sulfur' : sulfur_master, 'silicon' : silicon_master,
    'aluminium' : aluminium_master, 'niobium' : niobium_master, 'vanadium' : vanadium_master,
    'gauge' : gauge_master_list, 'width' : width_master_list, 'weight' : weight_master_list, 'location' : location_master_list,
    'defect': defect_master_list, "linear feat" : linfeet_master_list}


    ak_dict = pd.DataFrame(comprehensive_dict)
    
    ak_dict['defect_code'] = ak_dict.defect.str.extract('(\d+)')

    return ak_dict
    



