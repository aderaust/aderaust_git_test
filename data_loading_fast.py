#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import scipy.special

import plotly.tools as tls
# get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.graph_objs as go

import plotly as py
# py.offline.init_notebook_mode()



import dash 
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input

import plotly.figure_factory as ff

import dash_table


# In[2]:


# loading data

grades_final = pd.read_csv('msu_tidy_grades_final.csv', parse_dates = ['date'])

probability_lists = np.loadtxt('probability_lists.csv')

grades_final['probability_lists'] = list(probability_lists)


# removing courses with a perfect gpa 


grades_final = grades_final.dropna()


# ordering course_names alphabetically and then teachers alphabetically

grades_final = grades_final.sort_values(by = ['course_name', 'instructors'])


# In[3]:


time_series_grades = grades_final[['course_name', 'mean', 'date', 'instructors']]


# In[4]:


all_class_instruc = list(grades_final.groupby(['course_name', 'instructors']).probability_lists)


# In[5]:


# assigning value to possible gpa distributions 

gpa_dist = np.array([4, 3.5, 3, 2.5, 2, 1.5, 1, 0])


# In[6]:


# normalizing every course distribution


def normalize_gpa_dist(x):
    '''This funcion simply aggregates and normalizes a teacher's
    course distributions. If the same teacher taught the same course,
    it sums their respective distributions together, then divides by
    number of courses to normalize '''
    inital_normalized_dists = []
    for i in range(len(x)):
        number_of_courses = len(x[i][1])
        
        # summing distributions together
        
        distribution_sum = np.sum(x[i][1].values)
        
        
        # normalizng by dividing by the number of courses taught 
        
        
        normalized_dist = distribution_sum / number_of_courses
        
        inital_normalized_dists.append([x[i][0], normalized_dist])
        
    return inital_normalized_dists
        
all_normalized_dists = normalize_gpa_dist(all_class_instruc)


# In[7]:


# pulling each unique course as well as every teacher

teacher_list = []
for i in range(len(all_normalized_dists)):
    teacher_list.append(all_normalized_dists[i][0][1])
    
teacher_list = np.array(teacher_list)


course_list = []
for i in range(len(all_normalized_dists)):
    course_list.append(all_normalized_dists[i][0][0])
    
course_list = np.array(course_list)


# In[8]:


#  multi_teacher_index_test = np.where(teacher_list == ['A M SAEED'])


# target_teacher_prob_test = np.sum(np.array(all_normalized_dists)[multi_teacher_index_test][:,1])

# target_teacher_prob_test =  target_teacher_prob_test / len(multi_teacher_index_test[0])


# In[9]:


# # np.array(all_normalized_dists)[multi_teacher_index_test]
# import time

# dummy = np.repeat(.125, len(gpa_dist))



# start = time.time()
# # dummy_range = range(10000)
# # ranges = range(10000)
# # [np.random.rand() for i in range(10000)]



# boot_strapped_means = np.array([np.mean(
#     np.random.choice(gpa_dist, size = 100, p = dummy))
#                                 for i in range(10000)])




# end = time.time()

# end - start 


# ## standard deviation

# In[10]:


# std calculator 


def std_bootstrapper(teacher_input = None, course_input = None):
    '''This function  generates 100 random samples with replacement from a normalized distribution 
    it then calculates the mean then stores it in a array. 
    Finally, the mean of those means as well as the standard
    deviation is calculated in order to calculate a confidence
    Interval'''
    
    
    
    target_array = np.array(all_normalized_dists)
        
#     multi_teacher_index = np.where(teacher_list == str(teacher_input).upper())
    
#     multi_course_index = np.where(course_list == str(course_input).upper())
    
    
    
    # not entering a course nor a teacher
    if teacher_input is None  and course_input is None:
        return ValueError("Please enter at least a specific course or a specific teacher")


    
    
    # selecting to analyze a teacher
    elif teacher_input is not None and course_input is None:
#         teacher_input = teacher_input.upper()
        
        multi_teacher_index = np.where(teacher_list == teacher_input)

        
        target_teacher_prob = np.sum(target_array[multi_teacher_index][:,1])
        
        target_teacher_prob =  target_teacher_prob / len(multi_teacher_index[0])
        
        
#         target_teacher_prob = np.round(target_teacher_prob)
        
        boot_strapped_stds = np.array([np.std(
            np.random.choice(gpa_dist, size = 100, p = target_teacher_prob))
                                        for i in range(2500)])
        
        
        mean_of_std = np.mean(boot_strapped_stds)
        
#         sigma_of_bootstrap = np.std(boot_strapped_means)
        
#         ci_bootstrap = stats.norm.interval(0.95, loc=mean_of_bootstrap, scale=sigma_of_bootstrap)
        
        # returning courses offered by professor
        
#         list_teacher_courses = course_list[multi_teacher_index]
        
        return mean_of_std
        

        
        
        
    elif teacher_input is None and course_input is not None:
#         course_input = course_input.upper()
        
        multi_course_index = np.where(course_list == course_input)
        
        target_course_prob = np.sum(target_array[multi_course_index][:,1])
        
        target_course_prob =  target_course_prob / len(multi_course_index[0])
        
        
        boot_strapped_stds = np.array([np.std(
            np.random.choice(gpa_dist, size = 100, p = target_course_prob))
                                        for i in range(2500)])
        
        
        mean_of_std = np.round(np.mean(boot_strapped_stds), 2)
        
#         sigma_of_bootstrap = np.std(boot_strapped_means)
        
#         ci_bootstrap = stats.norm.interval(0.95, loc=mean_of_bootstrap, scale=sigma_of_bootstrap)
        
        # returning courses offered by professor
        
#         list_course_instructors = teacher_list[multi_course_index]
        
        return mean_of_std
        

        
        


# In[11]:


def multi_std_bootstrapper(teacher_input, course_input):
    
        target_array = np.array(all_normalized_dists)
        # finding index for specific course and teacher
        
        golden_list_of_stds = []
        
        for teacher in teacher_input:
        
            specific_course_and_teacher_index = np.where(
                (teacher_list == teacher) & (course_list == course_input))

            # pulling probability for specific course and teacher
            
            specific_course_and_teacher_prob = target_array[specific_course_and_teacher_index].flatten()[1]

            # generating random samples for specific course and teacher

            boot_strapped_stds = np.array([np.std(
                np.random.choice(gpa_dist, size = 100, p = specific_course_and_teacher_prob))
                                            for i in range(2500)])

            # math for calculating CI for specific course and teacher

            mean_of_std = np.round(np.mean(boot_strapped_stds), 2)



            # returning list of course_name & respective teacher

            
            golden_list_of_stds.append(mean_of_std)
        
        return golden_list_of_stds

multi_std_bootstrapper(['CHRISTIAN GOULDING'], 'FI_311')


# ## mean

# In[12]:


# this function returns the bootstrapped GPA for a given professor or for a given course

def mean_bootstrapper(teacher_input = None, course_input = None):
    '''This function  generates 100 random samples with replacement from a normalized distribution 
    it then calculates the mean then stores it in a array. 
    Finally, the mean of those means as well as the standard
    deviation is calculated in order to calculate a confidence
    Interval'''
    
    
    
    target_array = np.array(all_normalized_dists)
        
#     multi_teacher_index = np.where(teacher_list == str(teacher_input).upper())
    
#     multi_course_index = np.where(course_list == str(course_input).upper())
    
    
    
    # not entering a course nor a teacher
    if teacher_input is None  and course_input is None:
        return ValueError("Please enter at least a specific course or a specific teacher")


    
    
    # selecting to analyze a teacher
    elif teacher_input is not None and course_input is None:
#         teacher_input = teacher_input.upper()
        
        multi_teacher_index = np.where(teacher_list == teacher_input)

        
        target_teacher_prob = np.sum(target_array[multi_teacher_index][:,1])
        
        target_teacher_prob =  target_teacher_prob / len(multi_teacher_index[0])
        
        
#         target_teacher_prob = np.round(target_teacher_prob)
        
        boot_strapped_means = np.array([np.mean(
            np.random.choice(gpa_dist, size = 100, p = target_teacher_prob))
                                        for i in range(2500)])
        
        
        mean_of_bootstrap = np.mean(boot_strapped_means)
        
        sigma_of_bootstrap = np.std(boot_strapped_means)
        
        ci_bootstrap = stats.norm.interval(0.95, loc=mean_of_bootstrap, scale=sigma_of_bootstrap)
        
        # returning courses offered by professor
        
        list_teacher_courses = course_list[multi_teacher_index]
        
        return boot_strapped_means, ci_bootstrap, list_teacher_courses
        

        
        
        
    elif teacher_input is None and course_input is not None:
#         course_input = course_input.upper()
        
        multi_course_index = np.where(course_list == course_input)
        
        target_course_prob = np.sum(target_array[multi_course_index][:,1])
        
        target_course_prob =  target_course_prob / len(multi_course_index[0])
        
        
        boot_strapped_means = np.array([np.mean(
            np.random.choice(gpa_dist, size = 100, p = target_course_prob))
                                        for i in range(2500)])
        
        
        mean_of_bootstrap = np.mean(boot_strapped_means)
        
        sigma_of_bootstrap = np.std(boot_strapped_means)
        
        ci_bootstrap = stats.norm.interval(0.95, loc=mean_of_bootstrap, scale=sigma_of_bootstrap)
        
        # returning courses offered by professor
        
        list_course_instructors = teacher_list[multi_course_index]
        
        return boot_strapped_means, ci_bootstrap, list_course_instructors
        
        
        
        
        

        
    elif teacher_input is not None and course_input is not None:
#         teacher_input = teacher_input.upper()
#         course_input = course_input.upper()

        
        # finding index for specific course and teacher
        
        specific_course_and_teacher_index = np.where(
            (teacher_list == teacher_input) & (course_list == course_input))

        # pulling probability for specific course and teacher
        
        specific_course_and_teacher_prob = target_array[specific_course_and_teacher_index].flatten()[1]
        
        # generating random samples for specific course and teacher
        
        boot_strapped_means = np.array([np.mean(
            np.random.choice(gpa_dist, size = 100, p = specific_course_and_teacher_prob))
                                        for i in range(1000)])
        
        # math for calculating CI for specific course and teacher
        
        mean_of_bootstrap = np.mean(boot_strapped_means)
        
        sigma_of_bootstrap = np.std(boot_strapped_means)
        
        ci_bootstrap = stats.norm.interval(0.95, loc=mean_of_bootstrap, scale=sigma_of_bootstrap)
        
        # returning list of course_name & respective teacher
        
        list_teacher_and_course = [teacher_input, course_input]
        
        return boot_strapped_means, ci_bootstrap, list_teacher_and_course

        
        


# In[13]:


## creating function that can generate data of lists of multiple teachers


def multi_mean_bootstrapper(teacher_input, course_input):
    
        target_array = np.array(all_normalized_dists)
        # finding index for specific course and teacher
        
        golden_list_of_course_and_teachers = []
        
        for teacher in teacher_input:
        
            specific_course_and_teacher_index = np.where(
                (teacher_list == teacher) & (course_list == course_input))

            # pulling probability for specific course and teacher
            
            specific_course_and_teacher_prob = target_array[specific_course_and_teacher_index].flatten()[1]

            # generating random samples for specific course and teacher

            boot_strapped_means = np.array([np.mean(
                np.random.choice(gpa_dist, size = 100, p = specific_course_and_teacher_prob))
                                            for i in range(2500)])

            # math for calculating CI for specific course and teacher

            mean_of_bootstrap = np.mean(boot_strapped_means)

            sigma_of_bootstrap = np.std(boot_strapped_means)

            ci_bootstrap = stats.norm.interval(0.95, loc=mean_of_bootstrap, scale=sigma_of_bootstrap)

            # returning list of course_name & respective teacher

            list_teacher_and_course = [teacher, course_input]
            
            golden_list_of_course_and_teachers.append([boot_strapped_means, ci_bootstrap, list_teacher_and_course])
        
        return golden_list_of_course_and_teachers

    
    


# In[14]:


multi_mean_bootstrapper(['AYLIN ALIN', 'AARON C HENSLEY'], ['STT_315']) # lower bound CI

# function is working


# In[15]:


# mth_132_means, mth_132_ci, mth_132_teachers = mean_bootstrapper(course_input="MTH_132")
# eric_mth_132_means, eric_ci, eric_courses = mean_bootstrapper(teacher_input='ERICK A VERLEYE')


# In[16]:


# mth_132_fig = plt.figure() 

# sns.distplot(mth_132_means, hist = False, kde = True, norm_hist= True,
#             kde_kws = {'shade': True, 'linewidth': 3}, 
#                   label = 'All MTH 132')

# sns.distplot(eric_mth_132_means, hist = False, kde = True, norm_hist= True,
#             kde_kws = {'shade': True, 'linewidth': 3}, 
#                   label = 'MTH 132 Sec. 09')


# # plt.axvline(yang_ci[0], color = 'r', linestyle = "dashed")
# # plt.axvline(yang_ci[1], color = 'r', linestyle = "dashed")
# plt.xlabel("Bootstrapped Mean GPA")
# plt.yticks([])
# plt.title("MTH 132 Random Sampling Mean GPA")

# print("Random Sample MTH 132 Sec. 09 Mean GPA", np.round(np.mean(eric_mth_132_means),2))
# print("Random Sample All MTH 132 Mean GPA", np.round(np.mean(mth_132_means), 2))


# # sns.distplot(cmse_bootstrapped_means, hist = False, kde = True,
# #                  kde_kws = {'shade': True, 'linewidth': 3}, 
# #                   label = 'All CMSE')


# In[17]:


# # two sample ztest



# ztest ,pval1 = stests.ztest(
#     mth_132_means, x2=eric_mth_132_means,
#     value=0,alternative='smaller')

# print(float(pval1))
# if pval1<0.05:
#     print("reject null hypothesis")
# else:
#     print("accept null hypothesis")


# In[18]:


# ztest ,pval1 = stests.ztest(
#     np.array([100,100,100]), x2=[100,100,99],
#     value=0,alternative='two-sided')


# In[19]:


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(external_stylesheets= external_stylesheets)



app.config['suppress_callback_exceptions']=True


# In[20]:


course_list_show = np.unique(course_list)


# In[21]:


# making list of dictionaries for drop down menu
course_dic_list = []

for i, val in enumerate(course_list_show):
    course_dic_list.append({'label': val, 'value': val })
    
# course_dic_list = sorted(course_dic_list, key=lambda k: k['label']) 


# In[22]:


# loading all of msu_original Data

all_msu_bootstrap = np.load('all_msu_bootstrap.npy')

all_msu_bar = np.load('all_msu_gpa_bar.npy')

all_msu_ci = np.load('all_msu_ci.npy')


all_msu_mean = np.round(np.mean(all_msu_bootstrap), 3)

# all_msu_std = np.round(np.std(all_msu_bootstrap), 3)

# all_msu_std

all_msu_ratio = np.round(all_msu_mean / .878, 2)


# In[23]:


default_table_dic = {'Teacher': ["All MSU"],
                      'Rank: MAX(Mean / σ)' : [None],
                     'Simulated Mean GPA' : [all_msu_mean],
                     'Simulated σ GPA' : [.88],
                     '95% Mean GPA CI' : [(np.round(all_msu_ci[0], 2), " - ", np.round(all_msu_ci[1], 2))]
                    }


# In[24]:


# # creating default table
# default_table_dic = {'Teacher': ["All MSU CI"], 'Expected 5% Lower GPA' : [np.round(all_msu_ci[0], 2)],
#                      'Expected 5% Upper GPA' : [np.round(all_msu_ci[1], 2)]}

default_table_df = pd.DataFrame(default_table_dic)

# creating default/standard column labels
default_column_labels = [{"name": i, "id": i} for i in default_table_df.columns]

# [{"name": i, "id": i} for i in default_table_df.columns]


# In[25]:


# making default all msu histogram plot

hist_data = [all_msu_bootstrap]
group_labels = ['All MSU']

fig = ff.create_distplot(hist_data, group_labels, bin_size=.02, show_rug=False, show_hist=True)

fig['layout'].update(title='All MSU GPA Distribution')

fig['layout']['yaxis'].update(autorange=True,
        showgrid=True,
        zeroline=False,
        showline=False,
        ticks='',
        showticklabels=False)

fig['layout']['xaxis'].update(title='Bootstrapped Mean GPA')




# py.offline.iplot(fig)


# In[26]:



initial_time_graph = grades_final.loc[:, ['date', 'mean']].groupby(['date']).apply(np.mean)

x_all_msu_data = initial_time_graph.index

y_all_msu_data = initial_time_graph['mean'].values


all_msu_time_graph = [go.Scatter(x = x_all_msu_data, y = y_all_msu_data)]


all_msu_layout = {
'title': "All MSU",


        'xaxis' : { 'title' : 'Time'},
        'yaxis' : {'title' : 'Mean GPA'}
}


time_fig = go.Figure(all_msu_time_graph, all_msu_layout)



# py.offline.iplot(time_fig)


# ## description layout

# In[27]:


description_layout  = [html.H2('Purpose'),
                     html.Label('MSU Optimize allows you to easily compare GPA distributions among teachers and make competitive decisions for your GPA.'),
                     html.H2('Value Add'), html.H4('1. Exhaustive, Cumulative Distributions'),
                     html.Label('Since Fall of 2011, MSU has been required to record GPA distributions for every course offered at MSU.  In order to fully take advantage of this data, this application cumulates every probability distribution available for each unique professor and course.'),
                     html.H6('Example'),
                     html.Label('If John Smith taught MSU_101 in Fall of 2015  then again in Spring 2019, his GPA distributions will differ but be similar in nature.'),
                     html.Img(src = app.get_asset_url('ReadMe Fall Smith.png')), html.Img(src = app.get_asset_url('ReadMe Spring Smith.png')),
                     html.Img(src = app.get_asset_url('ReadMe Combined Smith.png')),
                     html.H4('2. Course Mean and Standard Deviation Estimation through Simulation'),
                     html.Label('The distributions above are interesting, but they do not offer any useful or practical statistical properties. What’s more, the difficulty of comparing such oddly shaped distributions for more and more instructors only compounds. '),
                     html.H6(''),
                     html.Label('Random sampling through Bootstrapping is a technique that allows one to generate many random samples from one instructor’s GPA distribution. This app randomly takes 100 samples 2,500 times from a teacher’s exhaustive GPA probability distribution, like above, then calculates the mean of these 2,500 samples to form a distribution of means. From the central limit theorem, this distribution will be approximately normal. This allows us to easily and visually compare the distributions of different instructors teaching the same course.'),
                     html.Img(src = app.get_asset_url('readme bootstrap.png')),
                     html.Label('As you can see from the plot, Smith’s bootstrapped mean GPA distribution for MSU_101 is approximately normal. The horizontal red lines represent a 95% confidence interval for where his true mean GPA lies. In laymen’s terms, this is roughly interpreted as “the mean GPA of 95% of the random samples from Smith’s MSU_101 original distribution will fall between 3.3 to 3.6”. This process of iteratively random sampling and calculating descriptive statistics can also be completed for other statistics such as standard deviation.'),
                     html.H6(''),
                     html.Label('To find the best estimate for a course’s standard deviation or sigma, one samples the exhaustive distribution above 2,500 times, just like above, but instead of calculating the mean each sample, one calculates the standard deviation 2,500 times. After 2,500 times the average standard deviation of all samples is the most accurate sigma or standard deviation.'),
                     html.H4('3. Optimization/Rank Methodology'),
                     html.Label('Putting these techniques together, one can be left with the best estimate for an instructors mean GPA and standard deviation. Taking the ratio of their mean and standard deviation allows one to take into account both parameters of a instructors GPA distribution when making a course decision. For instance, one instructor could have a higher historical mean, however, if they have a large standard deviation, the course could be considered not as attractive because of risk for a low grade. '),
                     html.H6(''),
                     html.Label('Below is a fictitious example of the rank ratio being put to use. The teacher in the second row is actually ranked above the teacher in the third row, even with a lower mean, because teacher two has a lower course mean standard deviation.'),
                     html.Img(src = app.get_asset_url('readme_ranktable.png'), width = '70%', height = '70%'),
                     html.H4('4. Course/Instructor Time-Series Analysis'),
                     html.Label('This dashboard also allows you to observe how overall courses and individual instructors mean GPAs have trended over each semester.')
                      ]


# ## Markdown

# In[28]:


# formatting app


app.layout = html.Div(id = "all_app", children =[
    html.H1("MSU Optimize"),
    html.Div([dcc.Tabs(id = 'tabs', children = [dcc.Tab(label = 'Course Analysis', children = [html.Label('Choose a Course to Analyze'),
    
    dcc.Dropdown(id = "course_input_dropdown",
    options = course_dic_list, style = dict(width = "68%")),
    
    
    html.Label("Choose Course's Teacher"),
    

    dcc.Dropdown(id = "teacher_input_dropdown", multi = True,
    options = [{'label': 'Select a Course', 'value': 'Select A Course'}], style = dict(width = "68%")),
    
    html.Button('New Course !',id='reset_button'),
    
   
    
    dcc.Graph(id = 'course_graph', figure = fig,
             config = {'staticPlot': True}),

    dash_table.DataTable(id = 'ci_table', data = default_table_df.to_dict('records'),
                        columns = default_column_labels, sorting=True, sorting_type="multi",
                            style_data_conditional=[{
        'if': {'column_id': 'Rank: MAX(Mean / σ)'},
        'backgroundColor': '#3D9970',
        'color': 'white',
    }]),
    
    
    dcc.Graph(id = 'time_series_graph', figure = time_fig,
             config = {'staticPlot': True})

                                                                           ]
                                              ),
                                      dcc.Tab(label = 'Description', children = description_layout)
                                     ]
            )])
]
                     )




# In[29]:



# updating initial course figure

# @app.callback(Output('course_graph', 'figure'),
#               [Input('course_input_dropdown', 'value')])



# def course_fig(update_value):
    
#     bs_data, bs_ci, meta_data = mean_bootstrapper(course_input= update_value)
    
    
# #     data = []
#     hist_data = [bs_data]
#     group_labels = [update_value]

#     fig = ff.create_distplot(hist_data, group_labels, bin_size=.01, show_rug=False, show_hist=True)

#     fig['layout'].update(title=update_value)

#     # turning of the yaxis variable
#     fig['layout']['yaxis'].update(autorange=True, showgrid=True, zeroline=False, showline=False, ticks='', showticklabels=False)

#     fig['layout']['xaxis'].update(title='Bootstrapped Mean GPA')

# #     data.append(fig) # data must be in list form
#     return  fig 


all_normalized_dists_easy = [all_normalized_dists[i][0] for i, val in enumerate(all_normalized_dists)]

all_normalized_dists_easy = np.array(all_normalized_dists_easy)


@app.callback(Output('teacher_input_dropdown', 'options'),
              [Input('course_input_dropdown', 'value')])






def fill_dropdown_menu(update_teacher_value):
#     update_teacher_value = update_teacher_value['layout']['title']['text']
    intermediate_teacher_index = np.where(all_normalized_dists_easy == update_teacher_value)[0]    
    # return objects of normalized dists where the course update value matches 
    
    teacher_dic_list = teacher_list[intermediate_teacher_index]
    
    teacher_dic_list = list(teacher_dic_list)
    

    final_teacher_dic_list = []

    for i, val in enumerate(teacher_dic_list):
        final_teacher_dic_list.append({'label': val, 'value': val })
    
    return final_teacher_dic_list




# In[30]:


@app.callback(
    Output('course_graph', 'figure'),
    [Input('teacher_input_dropdown', 'value'),
     Input('course_input_dropdown', 'value')])
#      Input('course_graph', 'figure')])

def multi_teacher_fig_bind(teacher_input_dropdown, course_input_dropdown):
    
    
    if teacher_input_dropdown is None:
    
    
        bs_data, bs_ci, meta_data = mean_bootstrapper(course_input = course_input_dropdown)

        relavent_teacher_data = [bs_data]
        relavent_teacher_labels = [course_input_dropdown]


        teach_fig = ff.create_distplot(relavent_teacher_data,
                                       relavent_teacher_labels, bin_size=.01, show_rug=False, show_hist=True)

        teach_fig['layout'].update(title = course_input_dropdown)

        # turning of the yaxis variable
        teach_fig['layout']['yaxis'].update(autorange=True,
                                            showgrid=True, zeroline=False, showline=False, ticks='',
                                            showticklabels=False)

        teach_fig['layout']['xaxis'].update(title='Bootstrapped Mean GPA')

    #     data.append(fig) # data must be in list form
        return  teach_fig

    
    
    # generating bootstrap means and CI for all teacher_inputs
    
    else:
    
    
        bs_data, bs_ci, meta_data = mean_bootstrapper(course_input = course_input_dropdown)

        relavent_teacher_data = [bs_data]
        relavent_teacher_labels = [course_input_dropdown]

    
        list_of_relavent_teacher = multi_mean_bootstrapper(teacher_input_dropdown, course_input_dropdown)
        
        
        
        
        for i in list_of_relavent_teacher:
            relavent_teacher_data.append(i[0])
            relavent_teacher_labels.append(i[2][0])


#         relavent_teacher_data.append(bs_data)
#         relavent_teacher_labels.append(course_input_dropdown)
        
            
        teach_fig = ff.create_distplot(relavent_teacher_data,
                                       relavent_teacher_labels, bin_size=.01, show_rug=False, show_hist=False)

        teach_fig['layout'].update(title= course_input_dropdown)

        # turning of the yaxis variable
        teach_fig['layout']['yaxis'].update(autorange=True,
                                            showgrid=True, zeroline=False, showline=False, ticks='',
                                            showticklabels=False)

        teach_fig['layout']['xaxis'].update(title='Bootstrapped Mean GPA')

    #     data.append(fig) # data must be in list form
        return  teach_fig 


# In[31]:



@app.callback(Output('all_app','children'),
             [Input('reset_button','n_clicks')])
def update(reset):
    
    if reset > 0:
        return [
    html.H1("MSU Optimize"),
    dcc.Tabs(id = 'tabs', children = [dcc.Tab(label = 'Course Analysis', children = [html.Label('Choose a Course to Analyze'),
    
    dcc.Dropdown(id = "course_input_dropdown",
    options = course_dic_list, style = dict(width = "68%")),
    
    
    html.Label("Choose Course's Teacher"),
    

    dcc.Dropdown(id = "teacher_input_dropdown", multi = True,
    options = [{'label': 'Select a Course', 'value': 'Select A Course'}], style = dict(width = "68%")),
    
    html.Button('New Course !',id='reset_button'),
    
   
    
    dcc.Graph(id = 'course_graph', figure = fig,
             config = {'staticPlot': True}),

    dash_table.DataTable(id = 'ci_table', data = default_table_df.to_dict('records'),
                        columns = default_column_labels, sorting=True, sorting_type="multi",
                            style_data_conditional=[{
        'if': {'column_id': 'Rank: MAX(Mean / σ)'},
        'backgroundColor': '#3D9970',
        'color': 'white',
    }]),
    
    
    dcc.Graph(id = 'time_series_graph', figure = time_fig,
             config = {'staticPlot': True})

                                                                           ]
                                              ),
                                      dcc.Tab(label = 'Description', children = description_layout)
                                     ]
            )
]
    
    


# In[32]:


@app.callback(
    Output('ci_table', 'data'),
    [Input('teacher_input_dropdown', 'value'),
     Input('course_input_dropdown', 'value')])



def multi_teacher_stat_data(teacher_input_dropdown, course_input_dropdown):
    
    # if you are only analyzing a course
    
    if teacher_input_dropdown is None:
    
    
        course_data, course_ci, meta_data = mean_bootstrapper(course_input = course_input_dropdown)

        
        
        course_std = std_bootstrapper(course_input= course_input_dropdown)
        
#         course_table = pd.DataFrame(course_ci)
        
#         course_table = course_table.to_dict()
        

    
    
    
    
    
    
    

        teacher_table_dic = [{'Teacher': [course_input_dropdown],
                             '95% Mean GPA CI' : [(np.round(course_ci[0], 2), " - ",
                              np.round(course_ci[1], 2))],
                              'Simulated Mean GPA' : [np.round(np.mean(course_data), 3)],
                              'Simulated σ GPA' : [np.round(course_std, 2)]
                             }]
    
    
#         teacher_table_dic = [{'Teacher': [course_input_dropdown],
#                               '95% Mean GPA' : [np.round(course_ci[0], 2)], " , ", np.round(course_ci[1], 2)}]

        
                
#         teacher_table_df = pd.DataFrame(teacher_table_dic)


#         teacher_table_df_final = teacher_table_df.to_dict('records')
        
        
        return  teacher_table_dic

    else:
        
        
        course_data, course_ci, meta_data = mean_bootstrapper(course_input = course_input_dropdown)

        course_std = std_bootstrapper(course_input = course_input_dropdown)
    
        list_of_relavent_teacher = multi_mean_bootstrapper(teacher_input_dropdown, course_input_dropdown)
        
        
        easy_stds = multi_std_bootstrapper(teacher_input_dropdown, course_input_dropdown)
        
        
        
        relavent_teacher_lower_ci = []
        relavent_teacher_upper_ci = []
        
        relavent_teacher_mean = []
        
        for i in list_of_relavent_teacher:
            relavent_teacher_lower_ci.append(i[1][0])
            relavent_teacher_upper_ci.append(i[1][1])
            relavent_teacher_mean.append(np.round(np.mean(i[0]), 3))
        
        # inserting the overall_course code, CI, overall mean, and standard deviations
        
        teacher_input_dropdown.insert(0, course_input_dropdown)
        
        relavent_teacher_lower_ci.insert(0, course_ci[0])
        relavent_teacher_upper_ci.insert(0, course_ci[1])
        relavent_teacher_mean.insert(0, np.round(np.mean(course_data), 3))
        
        easy_stds.insert(0, course_std)
        
        
        
        relavent_teacher_lower_ci = np.round(np.array(relavent_teacher_lower_ci), 2).tolist()
        
        relavent_teacher_upper_ci = np.round(np.array(relavent_teacher_upper_ci), 2).tolist()
    
        hyphen_format = np.repeat(" - ", len(relavent_teacher_lower_ci))
    
        teacher_ci_tuple = list(zip(relavent_teacher_lower_ci, hyphen_format, relavent_teacher_upper_ci))
        
        
        
        mean_std_ratio = np.array(relavent_teacher_mean) / np.array(easy_stds)
        
        sorted_mean_std_ratio = np.argsort(mean_std_ratio)[::-1].argsort() + 1
        
        final_ratio = sorted_mean_std_ratio.tolist()
        
        
        
        
        teacher_table_dic = {'Teacher': teacher_input_dropdown,
                             'Rank: MAX(Mean / σ)' : final_ratio,
                             '95% Mean GPA CI' : teacher_ci_tuple,
                             'Simulated Mean GPA' : relavent_teacher_mean,
                             'Simulated σ GPA' : easy_stds

                            }
        
#         teacher_table_dic = {'Teacher': teacher_input_dropdown,
#                             'Expected 5% Lower GPA' : relavent_teacher_lower_ci,
#                             'Expected 5% Upper GPA' : relavent_teacher_upper_ci}

        
        teacher_table_df = pd.DataFrame(teacher_table_dic)
        
        
        teacher_table_df.sort_values(by = ['Rank: MAX(Mean / σ)'], inplace=True) 
        
        
        
        final_teacher_dic = teacher_table_df.to_dict('records')

        return  final_teacher_dic 


# In[33]:


@app.callback(
    Output('time_series_graph', 'figure'),
    [Input('teacher_input_dropdown', 'value'),
     Input('course_input_dropdown', 'value')])

def time_mean_data_getter(teacher_inputs = None, course_input = None):
    '''This function retreives the appropriate mean data and plots the graph of mean gpa over time'''
    
    
    course_df = time_series_grades[time_series_grades['course_name'].isin([course_input])]
    
    xmin = np.min(course_df['date'])
    xmax = np.max(course_df['date'])
    
    
    horizontal_mean =  np.mean(course_df['mean'])

    
    # if only the course_input dropdown is selected 
    
    if teacher_inputs is None:
    
    # calculating overall mean gpa for all courses 
    
        
        # creating appropriate grouped df 

        grouped_df = course_df.loc[:, ['date', 'mean']].groupby(['date']).apply(np.mean)

        x_data = grouped_df.index

        y_data = grouped_df['mean'].values


        
        # plotting course mean overtime 
        
        data = [go.Scatter(x = x_data, y = y_data, name = course_input)]
        
        
        # adjusting the layout
        
        
        layout = {
                'title': course_input,
                'shapes': [
                    {  # Unbounded line at x = 4

                        'x0': np.min(x_data),
                        'y0': horizontal_mean,
                        'x1': np.max(x_data),
                        'y1': horizontal_mean,
                        'name': course_input + "Historical Average",
                        'line': {
                            'color': 'rgb(55, 128, 191)',
                            'width': 2,
                            'dash': 'dashdot'
                        }
                    },

                ],
                        'xaxis' : { 'title' : 'Time'},
                        'yaxis' : {'title' : 'Mean GPA'},
            }
        



        
        
        
        # creating figure 
        
        fig = go.Figure(data, layout)
        
        
        

        return fig
    
    # course input and teacher inputs start to fill out
    
    else:
        
        grouped_df = course_df.loc[:, ['date', 'mean']].groupby(['date']).apply(np.mean)

        x_all_data = grouped_df.index

        y_all_data = grouped_df['mean'].values


    
        
        all_course_data = go.Scatter(x = x_all_data, y = y_all_data, name = course_input)
        
        
        
        
        # filtering course-filtered dataframe based on the teacher inputs
        
        # quick if function that converts teacher inputs into a list so it will work with .isin() method
        
        if len(teacher_inputs) == 1:
            teacher_inputs = list(teacher_inputs)
        
        
        # filtering for teacher input

        course_and_teacher_df = course_df.loc[course_df['instructors'].isin(teacher_inputs)]

        # grouping by instructors then date, then calculating mean

        grouped_df = course_and_teacher_df.groupby(['instructors', 'date']).apply(np.mean)

        # for loop that loops through the teacher index and records the dates and respective course means

        data_list = [all_course_data]

        for teacher in grouped_df.index.levels[0]:

            current_teacher = grouped_df.loc[teacher]


            x_data = current_teacher.index

            y_data = current_teacher['mean']


            current_plot = go.Scatter(x=x_data, y=y_data, name=teacher)

            data_list.append(current_plot)

        # defining layout 
        
        layout = {
        'title': course_input,
#         'shapes': [
#             {  # Unbounded line at x = 4

#                 'x0': xmin,
#                 'y0': horizontal_mean,
#                 'x1': xmax,
#                 'y1': horizontal_mean,
#                 'name': course_input + "Historical Average",
#                 'line': {
#                     'color': 'rgb(55, 128, 191)',
#                     'width': 2,
#                     'dash': 'dashdot'
#                 }
#             },

#         ],
                'xaxis' : { 'title' : 'Time'},
                'yaxis' : {'title' : 'Mean GPA'}
    }
        
            
            
            
            
        fig = go.Figure(data_list, layout)

        return fig


        

    

        


# In[ ]:





# In[ ]:





# In[35]:


if __name__ == '__main__':
    app.run_server()


# In[ ]:




