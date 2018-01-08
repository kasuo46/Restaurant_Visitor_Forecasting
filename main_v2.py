import pandas as pd
from print_df_info import *
from weighted_mean import *
from sklearn.preprocessing import LabelEncoder
from rmsle import *
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

data_dir = 'Data/'
asi = pd.read_csv(data_dir + 'air_store_info.csv')
hsi = pd.read_csv(data_dir + 'hpg_store_info.csv')
sir = pd.read_csv(data_dir + 'store_id_relation.csv')
di = pd.read_csv(data_dir + 'date_info.csv')
ss = pd.read_csv(data_dir + 'sample_submission.csv')
avd = pd.read_csv(data_dir + 'air_visit_data.csv')
ar = pd.read_csv(data_dir + 'air_reserve.csv')
hr = pd.read_csv(data_dir + 'hpg_reserve.csv')

'''inner merge hpg_reserve, store_id_relation'''
hr_sir = pd.merge(hr, sir, on=['hpg_store_id'], how='inner')

'''for ar, convert objects to datetime, extract dates, calculate time diff'''
ar['visit_datetime'] = pd.to_datetime(ar['visit_datetime'])
ar['visit_date'] = ar['visit_datetime'].dt.date
ar['reserve_datetime'] = pd.to_datetime(ar['reserve_datetime'])
ar['reserve_date'] = ar['reserve_datetime'].dt.date
ar['reserve_visit_diff'] = ar['visit_datetime'] - ar['reserve_datetime']
ar['reserve_visit_diff'] = ar['reserve_visit_diff'].apply(lambda x: x.days)
# print_df_info(ar)

'''for ar, groupby sum air_store_id, visit_date'''
ar_sum = ar.groupby(['air_store_id', 'visit_date'], as_index=False)[['reserve_visit_diff', 'reserve_visitors']].sum()
ar_sum = ar_sum.rename(columns={'reserve_visit_diff': 'rvd_sum_ar', 'reserve_visitors': 'rv_sum_ar'})
# print_df_info(ar_sum)

'''for ar, groupby mean air_store_id, visit_date'''
ar_mean = ar.groupby(['air_store_id', 'visit_date'], as_index=False)[['reserve_visit_diff', 'reserve_visitors']].mean()
ar_mean = ar_mean.rename(columns={'reserve_visit_diff': 'rvd_mean_ar', 'reserve_visitors': 'rv_mean_ar'})
# print_df_info(ar_mean)

'''for ar, groupby count air_store_id, visit_date'''
ar_count = ar.groupby(['air_store_id', 'visit_date'], as_index=False)[['reserve_visit_diff', 'reserve_visitors']].count()
ar_count = ar_count.rename(columns={'reserve_visit_diff': 'rvd_count_ar', 'reserve_visitors': 'rv_count_ar'})
# print_df_info(ar_count)

'''for ar, inner merge sum, mean, count'''
ar_sum_mean = pd.merge(ar_sum, ar_mean, on=['air_store_id', 'visit_date'], how='inner')
ar_sum_mean_count = pd.merge(ar_sum_mean, ar_count, on=['air_store_id', 'visit_date'], how='inner')
# print_df_info(ar_sum_mean_count)

'''for hr_sir, convert objects to datetime, extract dates, calculate time diff'''
hr_sir['visit_datetime'] = pd.to_datetime(hr_sir['visit_datetime'])
hr_sir['visit_date'] = hr_sir['visit_datetime'].dt.date
hr_sir['reserve_datetime'] = pd.to_datetime(hr_sir['reserve_datetime'])
hr_sir['reserve_date'] = hr_sir['reserve_datetime'].dt.date
hr_sir['reserve_visit_diff'] = hr_sir['visit_datetime'] - hr_sir['reserve_datetime']
hr_sir['reserve_visit_diff'] = hr_sir['reserve_visit_diff'].apply(lambda x: x.days)
# print_df_info(hr_sir)

'''for hr_sir, groupby sum air_store_id, visit_date'''
hr_sir_sum = hr_sir.groupby(['air_store_id', 'visit_date'], as_index=False)[['reserve_visit_diff', 'reserve_visitors']].sum()
hr_sir_sum = hr_sir_sum.rename(columns={'reserve_visit_diff': 'rvd_sum_hr', 'reserve_visitors': 'rv_sum_hr'})
# print_df_info(hr_sir_sum)

'''for hr_sir, groupby mean air_store_id, visit_date'''
hr_sir_mean = hr_sir.groupby(['air_store_id', 'visit_date'], as_index=False)[['reserve_visit_diff', 'reserve_visitors']].mean()
hr_sir_mean = hr_sir_mean.rename(columns={'reserve_visit_diff': 'rvd_mean_hr', 'reserve_visitors': 'rv_mean_hr'})
# print_df_info(hr_sir_mean)

'''for hr_sir, groupby count air_store_id, visit_date'''
hr_sir_count = hr_sir.groupby(['air_store_id', 'visit_date'], as_index=False)[['reserve_visit_diff', 'reserve_visitors']].count()
hr_sir_count = hr_sir_count.rename(columns={'reserve_visit_diff': 'rvd_count_hr', 'reserve_visitors': 'rv_count_hr'})
# print_df_info(hr_sir_count)

'''for hr_sir, inner merge sum, mean'''
hr_sir_sum_mean = pd.merge(hr_sir_sum, hr_sir_mean, on=['air_store_id', 'visit_date'], how='inner')
hr_sir_sum_mean_count = pd.merge(hr_sir_sum_mean, hr_sir_count, on=['air_store_id', 'visit_date'], how='inner')
# print_df_info(hr_sir_sum_mean_count)

'''for avd, convert to datetime, add columns'''
avd['visit_date'] = pd.to_datetime(avd['visit_date'])
avd['visit_dow'] = avd['visit_date'].dt.dayofweek
avd['visit_year'] = avd['visit_date'].dt.year
avd['visit_month'] = avd['visit_date'].dt.month
avd['visit_date'] = avd['visit_date'].dt.date
# print_df_info(avd)

'''for ss, split, convert to datetime, add columns'''
ss['air_store_id'] = ss['id'].apply(lambda x: x[:-11])
ss['visit_date'] = ss['id'].apply(lambda x: x[-10:])
ss['visit_date'] = pd.to_datetime(ss['visit_date'])
ss['visit_dow'] = ss['visit_date'].dt.dayofweek
ss['visit_year'] = ss['visit_date'].dt.year
ss['visit_month'] = ss['visit_date'].dt.month
ss['visit_date'] = ss['visit_date'].dt.date
# print_df_info(ss)

''''''
unique_stores = ss['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'visit_dow': [i]*len(unique_stores)})
                    for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)
# print_df_info(stores)

'''for avd, groupby air_store_id, visit_dow, generate min, max, mean, median, count'''
avd_min = avd.groupby(['air_store_id', 'visit_dow'], as_index=False)['visitors'].min()
avd_min = avd_min.rename(columns={'visitors': 'min_visitors'})
stores = pd.merge(stores, avd_min, on=['air_store_id', 'visit_dow'], how='left')
avd_max = avd.groupby(['air_store_id', 'visit_dow'], as_index=False)['visitors'].max()
avd_max = avd_max.rename(columns={'visitors': 'max_visitors'})
stores = pd.merge(stores, avd_max, on=['air_store_id', 'visit_dow'], how='left')
avd_mean = avd.groupby(['air_store_id', 'visit_dow'], as_index=False)['visitors'].mean()
avd_mean = avd_mean.rename(columns={'visitors': 'mean_visitors'})
stores = pd.merge(stores, avd_mean, on=['air_store_id', 'visit_dow'], how='left')
avd_median = avd.groupby(['air_store_id', 'visit_dow'], as_index=False)['visitors'].median()
avd_median = avd_median.rename(columns={'visitors': 'median_visitors'})
stores = pd.merge(stores, avd_median, on=['air_store_id', 'visit_dow'], how='left')
avd_count = avd.groupby(['air_store_id', 'visit_dow'], as_index=False)['visitors'].count()
avd_count = avd_count.rename(columns={'visitors': 'count_visitors'})
stores = pd.merge(stores, avd_count, on=['air_store_id', 'visit_dow'], how='left')

'''make a tuple of longitude and latitude'''
asi['location'] = list(zip(asi['longitude'], asi['latitude']))
hsi['location'] = list(zip(hsi['longitude'], hsi['latitude']))

'''add air_store_info to stores'''
stores = pd.merge(stores, asi, how='left', on=['air_store_id'])
# print_df_info(stores)

'''for stores, change from / to space in air_genre_name, change from - to space in air_area_name'''
'''doesn't make sense to do this'''
# stores['air_genre_name'] = stores['air_genre_name'].apply(lambda x: x.replace('/', ' '))
# stores['air_area_name'] = stores['air_area_name'].apply(lambda x: x.replace('-', ' '))
# print_df_info(stores)

'''explore the genre, area, location'''
# ag_set = set(asi['air_genre_name'].unique())
# print(ag_set)
# aa_set = set(asi['air_area_name'].unique())
# print(len(aa_set))
# print(aa_set)
# al_set = set(asi['location'].unique())
# print(len(al_set))
# hg_set = set(hsi['hpg_genre_name'].unique())
# print(hg_set)
# ha_set = set(hsi['hpg_area_name'].unique())
# print(len(ha_set))
# print(ha_set)
# hl_set = set(asi['location'].unique())
# print(len(hl_set))
# print(ag_set.issubset(hg_set))
# print(aa_set.difference(ha_set))
# print(ha_set.difference(aa_set))
# print(hl_set == al_set)

'''modify hpg_genre_name to match with air_genre_name'''


'''for stores, encode the label for genre, area'''
stores['air_genre_name'] = LabelEncoder().fit_transform(stores['air_genre_name'])
stores['air_area_name'] = LabelEncoder().fit_transform(stores['air_area_name'])
# print_df_info(stores)

'''process di'''
di.rename(columns={'calendar_date': 'visit_date'}, inplace=True)
di['visit_date'] = pd.to_datetime(di['visit_date'])
di['day_of_week'] = di['visit_date'].dt.dayofweek
di['visit_date'] = di['visit_date'].dt.date
weekend_hol = di.apply((lambda x: (x.day_of_week == 6 or x.day_of_week == 5) and x.holiday_flg == 1), axis=1)
di.loc[weekend_hol, 'holiday_flg'] = 0
di['weight'] = ((di.index + 1) / len(di)) ** 5
# print_df_info(di, info=True, describe=True, head=True, lines=100)

'''build train and test set'''
train = pd.merge(avd, di, on=['visit_date'], how='left')
train.drop('day_of_week', axis=1, inplace=True)
# print_df_info(train)
test = pd.merge(ss, di, on=['visit_date'], how='left')
test.drop('day_of_week', axis=1, inplace=True)
# print_df_info(test)

'''merge stores'''
train = pd.merge(train, stores, on=['air_store_id', 'visit_dow'], how='left')
# print_df_info(train)
test = pd.merge(test, stores, on=['air_store_id', 'visit_dow'], how='left')
# print_df_info(test)

'''for train and test, merge ar, hr'''
train = pd.merge(train, ar_sum_mean_count, on=['air_store_id', 'visit_date'], how='left')
test = pd.merge(test, ar_sum_mean_count, on=['air_store_id', 'visit_date'], how='left')
train = pd.merge(train, hr_sir_sum_mean_count, on=['air_store_id', 'visit_date'], how='left')
test = pd.merge(test, hr_sir_sum_mean_count, on=['air_store_id', 'visit_date'], how='left')
# print_df_info(train, info=True)
# print_df_info(test, info=True)

'''add id to train'''
train['id'] = train['air_store_id'] + '_' + train['visit_date'].apply(lambda x: str(x))

'''for train and test, sum and average rvd_sum, rvd_mean, rv_sum, rv_mean'''
# print_df_info(train, info=True)
# print_df_info(test, info=True)
train['rv_sum'] = train[['rv_sum_ar', 'rv_sum_hr']].sum(axis=1)
train['rv_mean'] = train[['rv_mean_ar', 'rv_count_ar', 'rv_mean_hr', 'rv_count_hr']]\
    .apply(lambda x: weighted_mean(x['rv_mean_ar'], x['rv_count_ar'], x['rv_mean_hr'], x['rv_count_hr']), axis=1)
train['rvd_mean'] = train[['rvd_mean_ar', 'rvd_count_ar', 'rvd_mean_hr', 'rvd_count_hr']]\
    .apply(lambda x: weighted_mean(x['rvd_mean_ar'], x['rvd_count_ar'], x['rvd_mean_hr'], x['rvd_count_hr']), axis=1)
test['rv_sum'] = test[['rv_sum_ar', 'rv_sum_hr']].sum(axis=1)
test['rv_mean'] = test[['rv_mean_ar', 'rv_count_ar', 'rv_mean_hr', 'rv_count_hr']]\
    .apply(lambda x: weighted_mean(x['rv_mean_ar'], x['rv_count_ar'], x['rv_mean_hr'], x['rv_count_hr']), axis=1)
test['rvd_mean'] = test[['rvd_mean_ar', 'rvd_count_ar', 'rvd_mean_hr', 'rvd_count_hr']]\
    .apply(lambda x: weighted_mean(x['rvd_mean_ar'], x['rvd_count_ar'], x['rvd_mean_hr'], x['rvd_count_hr']), axis=1)
# print_df_info(train, info=True)
# print_df_info(test, info=True)

'''add date features, might be improved'''
train['date_int'] = train['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
test['date_int'] = test['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

'''add geo features, might be improved'''
train['var_max_lat'] = train['latitude'].max() - train['latitude']
train['var_max_long'] = train['longitude'].max() - train['longitude']
test['var_max_lat'] = train['latitude'].max() - test['latitude']
test['var_max_long'] = train['longitude'].max() - test['longitude']
train['long_plus_lat'] = train['longitude'] + train['latitude']
test['long_plus_lat'] = test['longitude'] + test['latitude']

'''encode air_store_id'''
le = LabelEncoder()
train['air_store_id_label'] = le.fit_transform(train['air_store_id'])
test['air_store_id_label'] = le.transform(test['air_store_id'])

# print_df_info(train, info=True, head=True, describe=True)
# print_df_info(test, info=True, head=True, describe=True)

'''save train and test to take a look'''
# train.to_csv('Data/train_save_1.csv')
# test.to_csv('Data/test_save_1.csv')

'''select the feature columns'''
feature_col = ['visit_dow', 'visit_year', 'visit_month', 'holiday_flg', 'min_visitors', 'max_visitors',
               'mean_visitors', 'median_visitors', 'count_visitors', 'air_genre_name', 'air_area_name',
               'latitude', 'longitude', 'rv_sum', 'rv_mean', 'rvd_mean', 'date_int', 'var_max_lat',
               'var_max_long', 'long_plus_lat', 'air_store_id_label', 'weight']

'''fillna, might be improved'''
# train = train.fillna(-1)
# test = test.fillna(-1)
for col in feature_col:
    train[col].fillna(train[col].mean(), inplace=True)
    test[col].fillna(train[col].mean(), inplace=True)
# print_df_info(train, info=True, describe=True)
# print_df_info(test, info=True, describe=True)

X = train[feature_col]
y = train['visitors']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1987)

'''GBR'''
model_gbr = GradientBoostingRegressor(learning_rate=0.2, verbose=True)
model_gbr.fit(X_train, y_train)
pred_train_gbr = np.clip(model_gbr.predict(X_train), a_min=0.0, a_max=None)
pred_val_gbr = np.clip(model_gbr.predict(X_val), a_min=0.0, a_max=None)

'''KNR'''
model_knr = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
model_knr.fit(X_train, y_train)
pred_train_knr = np.clip(model_knr.predict(X_train), a_min=0.0, a_max=None)
pred_val_knr = np.clip(model_knr.predict(X_val), a_min=0.0, a_max=None)

'''average of multiple models'''
pred_train_mix = (pred_train_gbr + pred_train_knr)/2
pred_val_mix = (pred_val_gbr + pred_val_knr)/2

'''fit and predict'''
print('GradientBoostingRegressor RMSLE: Train/Validation', rmsle(y_train, pred_train_gbr), rmsle(y_val, pred_val_gbr))
print('KNeighborsRegressor RMSLE: Train/Validation', rmsle(y_train, pred_train_knr), rmsle(y_val, pred_val_knr))
print('MixedRegressor RMSLE: Train/Validation', rmsle(y_train, pred_train_mix), rmsle(y_val, pred_val_mix))

'''prepare the submission file'''
# test['visitors'] = (model_gbr.predict(test[feature_col]) + model_knr.predict(test[feature_col])) / 2
# # test['visitors'] = model_knr.predict(test[feature_col])
# test['visitors'] = np.clip(test['visitors'].values, a_min=0.0, a_max=None)
# sub1 = test[['id', 'visitors']].copy()
# sub1.to_csv('Data/submission_3.csv', index=False)
