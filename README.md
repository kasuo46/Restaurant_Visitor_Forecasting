# Restaurant Visitor Forecasting
This project is to build a regression model on the reservation and visitation data to predict the total number of the visitors given a restaurant id and a date. The data needs to be merged and cleaned since it comes from two different sources. By using the extracted features, a regression model using GradientBoostingRegressor and KNeighborsRegressor in sklearn is built and achieved a rmsle score of 0.48.

## Introduction to the Dataset
The data is avaliable here https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/data. The data comes from two separate sites:
* Hot Pepper Gourmet (hpg): similar to Yelp, here users can search restaurants and also make a reservation online
* AirREGI / Restaurant Board (air): similar to Square, a reservation control and cash register system

The files in the dataset:
* `air_reserve.csv` This file contains reservations made in the air system.
* `hpg_reserve.csv` This file contains reservations made in the hpg system.
* `air_store_info.csv` This file contains information about select air restaurants.
* `hpg_store_info.csv` This file contains information about select hpg restaurants.
* `store_id_relation.csv` This file allows you to join select restaurants that have both the air and hpg system.
* `air_visit_data.csv` This file contains historical visit data for the air restaurants.
* `date_info.csv` This file gives basic information about the calendar dates in the dataset.

## Functions of Each File
* `print_df_info.py` prints the information of the dataframe.
* `rmsle.py` calculates rmsle score for the predictions and actual values.
* `weighted_mean.py` calcuates the weighted mean of two columns.
* `main_v2.py` is the main function. It performs data importing, merging, cleaning, generating features (date of week, dates, holiday, location, genre, history visitors and reservations, etc.), building regression models and measuring the performance.


## Possible Further Improvements
* Consider incorprating external data such as the weather, since weather is also a factor that impact visitors.
* Time-series analysis using LSTM.

## References
Thanks to the authors below who provide excellent kernels:

https://www.kaggle.com/the1owl/surprise-me

https://www.kaggle.com/headsortails/be-my-guest-recruit-restaurant-eda
