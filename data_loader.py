# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 10:41:47 2021

@author: Josh
"""

#imports
import dropbox
import pandas as pd
import numpy as np
#model imports
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy import stats
import statsmodels.api as sm
import warnings
import matplotlib.pyplot as plt
import os
#supress certain warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
pd.options.mode.chained_assignment = None

#local folder - this will need to be changed
save_to = r'final/'
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
##########################################
#   Data loading/manipulation section    #
##########################################
print("Beginning Data Loading...")
print(dname)
df = pd.read_csv('data\Zip_time_series.csv', converters = {'RegionName' : lambda x: str(x)})
zips = pd.read_csv('data/county_to_zip.csv', converters = {'ZIP' : lambda x: str(x)})
print("Data Loading Complete")
zip_cts = {}
seen_zips = []
print("Finding all unique Zip Codes for Counties")
for county in pd.unique(zips['COUNTYNAME']):
    
    county_df = zips[zips['COUNTYNAME'] == county]
    county_zips = [zipcode for zipcode in county_df['ZIP'] if zipcode not in seen_zips]
    seen_zips.extend([zipcode for zipcode in county_df['ZIP'] if zipcode not in seen_zips])
    zip_cts[county] = county_zips
print("Complete")
zip_sts = {}
seen_states = []
print("Finding all unique Zip Codes for States")
for state in pd.unique(zips['STATE']):
    
    state_df = zips[zips['STATE'] == state]
    state_zips = [zipcode for zipcode in state_df['ZIP'] if zipcode not in seen_states]
    seen_states.extend([zipcode for zipcode in state_df['ZIP'] if zipcode not in seen_states])
    zip_sts[state] = state_zips
print("Complete")
zip_cities = {}
seen_cities = []
print("Finding all unique Zip Codes for Cities")
for city in pd.unique(zips['CITY']):
    
    city_df = zips[zips['CITY'] == city]
    city_zips = [zipcode for zipcode in city_df['ZIP'] if zipcode not in seen_cities]
    seen_cities.extend([zipcode for zipcode in city_df['ZIP'] if zipcode not in seen_cities])
    zip_cities[city] = city_zips
print("Complete")
zips_unq = pd.DataFrame()
print("Creating Zip/County Combinations")
for key, val in zip_cts.items():
    
    for value in val:
        
        county_zip = pd.DataFrame(data = [key, value]).T
        county_zip.columns = ['County', 'ZipCode']
        zips_unq = zips_unq.append(county_zip)
print("Creating Zip/State Combinations")
print("Complete")
zips_unq_st = pd.DataFrame()
for key, val in zip_sts.items():
    
    for value in val:
        
        state_zip = pd.DataFrame(data = [key, value]).T
        state_zip.columns = ['State', 'ZipCode']
        zips_unq_st = zips_unq_st.append(state_zip)

zips_unq_st2 = zips_unq_st.groupby(['State', 'ZipCode']).size().reset_index().drop(0, axis = 1)
print("Complete")
print("Creating Zip/City Combinations")  
zips_unq_city = pd.DataFrame()
for key, val in zip_cities.items():
    
    for value in val:
        
        city_zip = pd.DataFrame(data = [key, value]).T
        city_zip.columns = ['City', 'ZipCode']
        zips_unq_city = zips_unq_city.append(city_zip)
print("Complete")
zips_unq_city2 = zips_unq_city.groupby(['City', 'ZipCode']).size().reset_index().drop(0, axis = 1)
#getting county/state/city data
df1 = df.set_index('RegionName').join(zips_unq.set_index('ZipCode'))
df2 = df1.join(zips_unq_st2.set_index('ZipCode'))
df3 = df2.join(zips_unq_city2.set_index('ZipCode'))

df4 = df3.sort_values(by = 'Date')
df5 = df4.reset_index().rename(columns = {'index' : 'Zip'})
date_missing = df5.groupby('Date').apply(lambda x: x.isnull().sum())
date_total = (df5.groupby('Date').size())
date_missing_pct = date_missing.div(date_total, axis = 0)
#want greater than 2010
df_want = df5[df5['Date'] >= "2010-01-01"]
#df_fill = df_want.groupby(['Date', 'County']).apply(lambda x: x.fillna(x.mean()))
df = pd.DataFrame()
print("Filling Missing Data")
for date in pd.unique(df_want['Date']):
    print('Filling data for: {}'.format(date))
    df_fill = df_want[df_want['Date'] == date]
    county_state_city = df_fill.loc[:, ['County', 'State', 'City', 'Zip']]
    #fill by county
    mean_values = df_fill.groupby(['Date', 'County']).apply(lambda x: x.fillna(x.mean())).reset_index(drop = True)
    #now fill leftover by state
    mean_values_state = mean_values.groupby(['Date', 'State']).apply(lambda x: x.fillna(x.mean())).reset_index(drop = True)
    #fill all leftovers by actual means
    mean_values_final = mean_values_state.groupby('Date').apply(lambda x: x.fillna(x.mean())).reset_index(drop = True)
    df = df.append(mean_values_final.reset_index(drop = True))
df['Zip'] = df['Zip'].astype(np.int64)
print("Complete")
##########################################
#           Modeling Section             #
##########################################
print("Begin Modeling")
#df = pd.read_csv('housing_full.csv')
#df['Zip'] = df['Zip'].astype(np.int64)
df_subset = df[['Date', 'Zip', 'InventorySeasonallyAdjusted_AllHomes',
             'MedianListingPrice_AllHomes', 'PctOfHomesIncreasingInValues_AllHomes',
             'ZHVI_AllHomes']]

def ols_timeshift(zipcode, df, plots=False):

    # filter full dataset for zipcode of interest, drop null rows
    
    df = df[df['Zip'] == zipcode]
    df.dropna(axis=0, inplace=True)
    
    # drop columns not used in regression model
    
    df_subset = df[['Date', 'Zip', 'InventorySeasonallyAdjusted_AllHomes',
             'MedianListingPrice_AllHomes', 'PctOfHomesIncreasingInValues_AllHomes',
             'ZHVI_AllHomes']]
    
    df_shift = df_subset.copy()
    
    df_date = df[['Date']]
    
    # set up lists that will contain correlation values for each time shift (1-6 months)
    
    season_corrs = []
    zhvi_corrs = []
    pct_corrs = []
    
    # loop through time shifts and append correlation values to lists
    
    for i in range(1, 7):
        df_corrtest = df_subset.copy()
        
        df_corrtest['InventorySeasonallyAdjusted_AllHomes'] = df_corrtest['InventorySeasonallyAdjusted_AllHomes'].shift(i)
        season_corrs.append(stats.pearsonr(df_corrtest['MedianListingPrice_AllHomes'][i:], 
                                           df_corrtest['InventorySeasonallyAdjusted_AllHomes'][i:]))
        
        df_corrtest['PctOfHomesIncreasingInValues_AllHomes'] = df_corrtest['PctOfHomesIncreasingInValues_AllHomes'].shift(i)
        pct_corrs.append(stats.pearsonr(df_corrtest['MedianListingPrice_AllHomes'][i:], 
                                           df_corrtest['PctOfHomesIncreasingInValues_AllHomes'][i:]))
        
        df_corrtest['ZHVI_AllHomes'] = df_corrtest['ZHVI_AllHomes'].shift(i)
        zhvi_corrs.append(stats.pearsonr(df_corrtest['MedianListingPrice_AllHomes'][i:], 
                                         df_corrtest['ZHVI_AllHomes'][i:]))
        
    # find the time shift for each predictor variable that results in best correlation
    # best time shift will be equal to list index + 1 because of 0-indexing
    # some correlations are negative, in which case the min correlation is found
    
    season_shift = season_corrs.index(min(season_corrs))+1
    zhvi_shift = zhvi_corrs.index(max(zhvi_corrs))+1
    pct_shift = pct_corrs.index(max(pct_corrs))+1
    
        
    # shift dataset using optimal time shift settings found in previous step

    df_subset['InventorySeasonallyAdjusted_AllHomes']=df_subset['InventorySeasonallyAdjusted_AllHomes'].shift(season_shift)
    df_subset['ZHVI_AllHomes']=df_subset['ZHVI_AllHomes'].shift(zhvi_shift)
    df_subset['PctOfHomesIncreasingInValues_AllHomes']=df_subset['PctOfHomesIncreasingInValues_AllHomes'].shift(pct_shift)
    
    # subset dataframe into X (independent variables) and Y (dependent variable)
    # also need to account for the shifted variables to make sure dfX and dfY are the same length, so
    # the first x number of rows are removed where x is the max time shift from the 3 independent variables
    
    dfX = df_subset[['InventorySeasonallyAdjusted_AllHomes', 'ZHVI_AllHomes' ,'PctOfHomesIncreasingInValues_AllHomes']][max(season_shift, zhvi_shift, pct_shift):]

    dfY = df_subset[['MedianListingPrice_AllHomes']][max(season_shift, zhvi_shift, pct_shift):]
    
    # split into train/test
    
    length = len(dfX)
    train = round(length*0.8)
    test = length-train

    dfX_train = dfX[:train]
    dfX_test = dfX[-test:]
    dfY_train = dfY[:train]
    dfY_test = dfY[-test:]

    
    # regression model

    regr=sm.OLS(dfY_train, dfX_train).fit()
    
    # predicted values

    y_pred = regr.predict(dfX_test)
    
    dfY_test['predicted'] = y_pred
    
    # return predicted values for insertion into original dataframe
    # indices are preserved for the eventual join

    modeleval = pd.DataFrame()
    modeleval['prediction'] = y_pred
    modeleval['test'] = dfY_test['MedianListingPrice_AllHomes'].values
    modeleval['zhvi'] = dfX_test['ZHVI_AllHomes']

    # optional plot generator
    
    if plots==True:

        modeleval = modeleval.join(df_date)
        plt.scatter(x=modeleval['Date'], y=modeleval['prediction'], label='Predicted')
        plt.plot(modeleval['Date'], modeleval['prediction'])
        plt.scatter(x=modeleval['Date'], y=modeleval['test'], label='Actual')
        plt.plot(modeleval['Date'], modeleval['test'])
        plt.scatter(x=modeleval['Date'], y=modeleval['zhvi'], label='ZHVI')
        plt.plot(modeleval['Date'], modeleval['zhvi'])
        plt.xticks(rotation=90)
        plt.ylabel('Home Price')
        plt.title('Predicted home prices for zip code {}'.format(zipcode))
        plt.grid(which='major', axis='both', alpha=0.5)
        plt.legend()
        plt.show()
        
    # find max time shift from above
 
    shift = max(season_shift, zhvi_shift, pct_shift)
    
    # shift all dependent variables using copy of original dataset
    
    dfX_shift = df_shift[['InventorySeasonallyAdjusted_AllHomes',  'ZHVI_AllHomes', 'PctOfHomesIncreasingInValues_AllHomes']][-shift:]
    
    # depending on max time shift, can use model to forecast into future months
    # e.g. if variables for this zip code were shifted 6 months, we can take
    # the last 6 months' worth of data in the original dataset and project
    # 6 months into the future due to the time lag
    
    forecast_pred = regr.predict(dfX_shift)
    
    forecast_length = len(forecast_pred)
    
    # list of future dates that will be sliced depending on timeshift (6 months is maximum)
    
    date_str = ['2018-01-31', '2018-02-28', '2018-03-31', '2018-04-30', '2018-05-31', '2018-06-30']

    # create forecast dataframe containing forecasted values for this zipcode
    
    forecast_df = pd.DataFrame()
    forecast_df['Date'] = date_str[:shift]
    forecast_df['Zip'] = zipcode
    forecast_df['Predicted'] = forecast_pred.values


    return dfY_test['predicted'], forecast_df


df_pred = pd.Series()
df_forecast = pd.DataFrame()
inc = 0

# NOTE this cell can take ~30min or more to run

for code in df['Zip'].unique():
    try:
        y, f = ols_timeshift(code, df_subset)
        df_pred = pd.concat([df_pred, y])
        df_forecast = pd.concat([df_forecast, f])
        inc += 1
    except:
        print(code)
        inc += 1
    
    # progress indicator
        
    if inc%100 == 0:
        print('{}% Complete'.format(round(inc/len(df['Zip'].unique()), 2)*100))

index = pd.read_csv('index.csv')['0']
print("Modeling Complete")
df_pred = pd.DataFrame(df_pred)
df_pred.index = index
df = df.join(df_pred)
df.rename(columns={0:"Predicted"}, inplace=True)
if not os.path.exists(dname + save_to):
    os.makedirs(dname + save_to)
df.to_csv(save_to + 'output_with_predictions.csv', index=False, header=True)
df_forecast.to_csv(save_to + 'forecasted_predictions.csv', index=False, header=True)

#df_end.to_csv('F:/GATech/SP21/CSE6242/Project/housing_full.csv', index = False)
#housing_full = pd.read_csv('F:/GATech/SP21/CSE6242/Project/housing_full.csv')
#finally, replace any leftover missing values by mean value overall.
#df_end2 = df_end.apply(lambda x: x.fillna(x.mean()), axis = 0)
