import pandas as pd

data_dir = 'Data/'

air_store_info = pd.read_csv(data_dir + 'air_store_info.csv')
# print(air_store_info.head())
# print(air_store_info.info())
# print(len(air_store_info['air_genre_name'].unique()))
# print(len(air_store_info['air_area_name'].unique()))

hpg_store_info = pd.read_csv(data_dir + 'hpg_store_info.csv')
# print(hpg_store_info.head())
# print(hpg_store_info.info())
# print(len(hpg_store_info['hpg_genre_name'].unique()))
# print(len(hpg_store_info['hpg_area_name'].unique()))

store_id_relation = pd.read_csv(data_dir + 'store_id_relation.csv')
# print(store_id_relation.head())
# print(store_id_relation.info())

date_info = pd.read_csv(data_dir + 'date_info.csv')
# print(date_info.head())
# print(date_info.info())
# print(date_info['day_of_week'].unique())
# print(date_info['holiday_flg'].unique())
# print(date_info.groupby('day_of_week')['calendar_date'].nunique())
# print(date_info.groupby('holiday_flg')['calendar_date'].nunique())

'''change calendar_date from str to datetime.date'''
date_info['calendar_date'] = pd.to_datetime(date_info['calendar_date']).dt.date

sample_submission = pd.read_csv(data_dir + 'sample_submission.csv')
# print(sample_submission.head())
# print(sample_submission.info())

'''split sample_submission (test) into store_id and date'''
sample_submission['air_store_id'] = sample_submission['id'].apply(lambda x: x[:-11])
sample_submission['date'] = sample_submission['id'].apply(lambda x: x[-10:])
# print(sample_submission.head())
# print(sample_submission.info())

'''in the test set, see if all the given air_store_id is in air_store_info'''
# set_sample_submission = set(sample_submission['air_store_id'].unique())
# set_air_store_info = set(air_store_info['air_store_id'].unique())
# print(len(set_sample_submission), len(set_air_store_info))
# print(set_sample_submission.issubset(set_air_store_info))

air_visit_data = pd.read_csv(data_dir + 'air_visit_data.csv')
# print(air_visit_data.head())
# print(air_visit_data.info())
# print(air_visit_data.describe())

'''transform visit_date in air_visit_data to datetime.date'''
air_visit_data['visit_date'] = pd.to_datetime(air_visit_data['visit_date']).dt.date
# print(air_visit_data.info())
# print(air_visit_data.shape)
# print(type(air_visit_data['air_store_id'][0]))
# print(type(air_visit_data['visit_date'][0]))
# print(air_visit_data.describe())

air_reserve = pd.read_csv(data_dir + 'air_reserve.csv')
# print(air_reserve.head())
# print(air_reserve.info())
# print(air_reserve.describe())

'''check the sets of air_store_id in air_visit_data, air_reserve'''
# set_air_store_id_visit = set(air_visit_data['air_store_id'].unique())
# set_air_store_id_reserve = set(air_reserve['air_store_id'].unique())
# print(len(set_air_store_id_visit), len(set_air_store_id_reserve))
# print(set_air_store_id_visit.issubset(set_air_store_id_reserve))
# print(set_air_store_id_reserve.issubset(set_air_store_id_visit))

'''add visit_date and reserve date columns to air_reserve'''
air_reserve['visit_datetime'] = pd.to_datetime(air_reserve['visit_datetime'])
air_reserve['visit_date'] = air_reserve['visit_datetime'].dt.date
air_reserve['reserve_datetime'] = pd.to_datetime(air_reserve['reserve_datetime'])
air_reserve['reserve_date'] = air_reserve['reserve_datetime'].dt.date
# print(air_reserve.head())
# print(air_reserve.info())
# print(air_reserve.shape)
# print(type(air_reserve['air_store_id'][0]))
# print(type(air_reserve['visit_date'][0]))
# print(air_reserve.describe())

'''merge air_visit_data, air_reserve'''
air_visit_reserve_data = pd.merge(air_visit_data, air_reserve, on=['air_store_id', 'visit_date'],
                                  how='outer')
# air_visit_reserve_data = air_visit_reserve_data[air_visit_reserve_data['_merge'] == 'left_only']
# print(air_visit_reserve_data.head(100))
# print(air_visit_reserve_data.info())
# print(air_visit_reserve_data.shape)
# print(air_visit_reserve_data.describe())
# air_visit_reserve_data.to_csv('Data/air_visit_reserve_data_outer_left_only.csv')

hpg_reserve = pd.read_csv(data_dir + 'hpg_reserve.csv')
# print(hpg_reserve.head())
# print(hpg_reserve.info(null_counts=True))
# print(hpg_reserve.describe())

'''add visit_date and reserve date columns to hpg_reserve'''
hpg_reserve['visit_datetime'] = pd.to_datetime(hpg_reserve['visit_datetime'])
hpg_reserve['visit_date'] = hpg_reserve['visit_datetime'].dt.date
hpg_reserve['reserve_datetime'] = pd.to_datetime(hpg_reserve['reserve_datetime'])
hpg_reserve['reserve_date'] = hpg_reserve['reserve_datetime'].dt.date

'''merge hpg_reserve, hpg_store_info'''
hpg_reserve_store_info = pd.merge(hpg_reserve, hpg_store_info, on='hpg_store_id', how='left')
# print(hpg_reserve_store_info.head())
# print(hpg_reserve_store_info.info(null_counts=True))

'''merge hgp_reserve_store_info, date_info on reserve_date'''
hpg_reserve_store_info_date_info = pd.merge(hpg_reserve_store_info, date_info,
                                            left_on='reserve_date', right_on='calendar_date',
                                            how='left').drop('calendar_date', axis=1)
# print(hpg_reserve_store_info_date_info.head(20))

