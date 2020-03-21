# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 00:28:27 2020

@author: Ted
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.linear_model import LinearRegression

class disease_data:
    def __init__(self):
        self.df = self.get_data()
        
    def get_data(self):
        # get data from kaggle
        # to connect to the API:
        # follow instructions here: https://www.kaggle.com/docs/api
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset='sudalairajkumar/novel-corona-virus-2019-dataset', unzip=True)
        
        # create a dataframe from the main data set  
        df = pd.read_csv('covid_19_data.csv')
        return df
       
class country_class():
    def __init__(self,df,country):
        self.df = df
        self.country = country
        self.country_df, self.country_recent = self.get_country_data()
        self.index_today = self.get_test_index_value()
        
    def get_country_data(self):   
        # get country-relevant data
        country_df = self.df.loc[self.df['Country/Region'] == self.country].reset_index(drop=True)
        
        # select the most recent data that follows an exponential increase
        country_recent = country_df.copy()
        country_recent['ObservationDate'] = pd.to_datetime(country_recent['ObservationDate'])
        # get data from the last 5 days
        today = datetime.today()
        delta = timedelta(days=6)
        diff = today - delta
        
        country_recent = country_recent.loc[country_recent.ObservationDate > diff]
        
        # cluster data by date to account for rows from different states/provinces
        country_recent = country_recent.groupby('ObservationDate').sum() 
        country_recent.reset_index(inplace=True)
        return country_df, country_recent
                   
    def get_test_index_value(self):
        # create an X variable to use for prediction
        # normally it would be simply the date
        # here, we find the difference between today's date and the
        # last date on the dataframe to create a new "index" value
        
        last_day = datetime.strptime(self.country_df['ObservationDate'].iloc[-1], '%m/%d/%Y')
        today = datetime.today()   
        diff = today - last_day    
        days = diff.days    
        input_val = self.country_recent.index[-1] + days   
        index_today = np.array([[input_val]])
        return index_today
    
class modeling():
    def __init__(self, variable, country_recent):
        self.country_recent = country_recent
        self.variable = variable
        self.model, self.cases, self.dates, self.r_squared = self.build_models()

        
    def build_models(self):
        # select only the confirmed cases
        variable = self.variable
        country_recent = self.country_recent.loc[:,[variable]]
        
        # set the 'dates' variable to be equal to the index vaues
        dates = np.array([country_recent.index]).transpose()
        
        # set cases equal to the nat-log of cases
        cases = np.array(np.log(country_recent[variable]))
        
        # fit the model
        reg = LinearRegression().fit(dates,cases)
        r_squared = reg.score(dates,cases)
        return reg, cases, dates,r_squared

def plot_data_and_predictions(country, variable, country_model, index_today):  
       
    # predict new data to plot on the graph
    x_new = np.linspace(0, max(country_model.dates), 100)
    y_new = country_model.model.predict(x_new)    
    
    # plot the data

    plt.scatter(country_model.dates,country_model.cases)
    plt.plot(x_new,y_new)
    plt.title('{} cases in {}'.format(variable, country))
    plt.ylabel('natural log of {} cases'.format(variable))
    plt.xlabel('previous {} days'.format(int(country_model.dates[-1])+1))
    plt.figtext(0.6, 0.4,'R-squared = %0.2f' % country_model.r_squared)
    plt.show()
    
   
    # print the predicted cases for the next three days
    # using the "index" value from above
    print("Predicted {} cases in {} for the next 3 days: {}, {}, {}".format(variable, 
                                                                            country,
                                                                            int(np.exp(country_model.model.predict(index_today))[0]),
                                                                            int(np.exp(country_model.model.predict(index_today+1))[0]),
                                                                            int(np.exp(country_model.model.predict(index_today+2))[0])))
def driver_function():
    full_data = disease_data()
    # list the countries of interest
    countries = ['Germany', 'US', 'Iran']
    variables = ['Confirmed', 'Deaths']
    for country in countries:
        country_info = country_class(full_data.df, country)

        for variable in variables:
            country_model = modeling(variable, country_info.country_recent)
            plot_data_and_predictions(country, 
                                      variable, 
                                      country_model, 
                                      country_info.index_today)
            
driver_function()
