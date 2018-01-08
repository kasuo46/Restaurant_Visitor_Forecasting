import pandas as pd
from sklearn.preprocessing import LabelEncoder
import datetime
import numpy as np

# data_dir = 'Data/'
#
# hpg_reserve = pd.read_csv(data_dir + 'hpg_reserve.csv')
# # print(hpg_reserve.head())
#
# date_info = pd.read_csv(data_dir + 'date_info.csv')
# print(date_info.head())
# print(date_info.info())
#
# hpg_reserve['visit_date'] = pd.to_datetime(hpg_reserve['visit_datetime']).dt.date
# hpg_reserve['reserve_date'] = pd.to_datetime(hpg_reserve['reserve_datetime']).dt.date
# print(hpg_reserve.head())
# print(hpg_reserve.info(null_counts=True))

# a = pd.DataFrame({'air_store_id': ['a1', 'b2', 'c3', 'd4', 'e5'],
#                   'visit_date': ['2017-01-01', '2017-01-02', '2017-01-03', '2017-01-04', '2017-01-05'],
#                   'visitor_a': [1, 2, 3, 4, 5]})
# print(a)
#
# b = pd.DataFrame({'air_store_id': ['a1', 'b2', 'h3', 'f6', 'g7'],
#                   'visit_date': ['2017-01-01', '2017-01-10', '2017-01-03', '2017-01-06', '2017-01-07'],
#                   'visitor_b': [1, 2, 3, 6, 7]})
# print(b)
#
# abi = pd.merge(a, b, on=['air_store_id', 'visit_date'], how='inner')
# print(abi)
#
# abo = pd.merge(a, b, on=['air_store_id', 'visit_date'], how='outer')
# print(abo)
#
# abl = pd.merge(a, b, on=['air_store_id', 'visit_date'], how='left')
# print(abl)
#
# abr = pd.merge(a, b, on=['air_store_id', 'visit_date'], how='right')
# print(abr)
#
# # (252108, 3)
# # (92378, 6)
# # inner (87181, 7)
# # outer (316422, 7)
# # left (311225, 7)
# # right (92378, 7)
# print(316422 - 87181 - (92378 - 87181))

# a = pd.DataFrame({'air_store_id': ['a1', 'b2', 'c3', 'd4', 'e5'],
#                   'visit_date': ['2017-01-01', '2017-01-02', '2017-01-03', '2017-01-04', '2017-01-05'],
#                   'visitor_a': [1, 2, 3, 4, 5]})
# print(a)
# a['visit_date'] = a['visit_date'].apply(lambda x: x.replace('-', '*'))
# print(a)
# a = pd.DataFrame({'air_store_id': ['a1', 'b2', 'a1', 'd4', 'e5'],
#                   'visit_date': ['2017-01-01', '2017-01-02', '2017-01-03', '2017-01-04', '2017-01-05'],
#                   'visitor_a': [1, 2, 3, 4, 5]})
# le = LabelEncoder()
# x = le.fit_transform(a['air_store_id'])
# print(type(x))
# print(x)
# d = datetime.datetime(2017, 12, 20)
# print(d.weekday())
# le = LabelEncoder()
# le.fit(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
# print(le.transform(['Tuesday', 'Friday', 'Friday']))
# a = pd.DataFrame({'air_store_id': ['a1', 'b2', 'c3', 'd4', 'e5'],
#                   'visit_date': ['2017-01-01', '2017-01-02', '2017-01-03', '2017-01-04', '2017-01-05'],
#                   'visitor_1': [1, 2, 3, 4, 5],
#                   'visitor_2': [2, 5, np.nan, 8, 9]})
# a['visitor_sum'] = a[['visitor_1', 'visitor_2']].mean(axis=1)
# a['air_store_id_visitor'] = a['air_store_id'] + a['visitor_1'].apply(lambda x: str(x))
# print(a)
# a = pd.DataFrame({'air_store_id': ['a1', 'b2', 'c3', 'd4', 'e5'],
#                   'visit_date': ['2017-01-01', '2017-01-02', '2017-01-03', '2017-01-04', '2017-01-05'],
#                   'mean_1': [np.nan, 2, 3, np.nan, 5],
#                   'ct_1': [np.nan, 6, 8, np.nan, 12],
#                   'mean_2': [2, np.nan, np.nan, np.nan, 9],
#                   'ct_2': [4, np.nan, np.nan, np.nan, 7]})
# print(a)
#
#
# def weighted_mean(mean1, ct1, mean2, ct2):
#     if (ct1 == 0 or np.isnan(ct1)) and (ct2 == 0 or np.isnan(ct2)):
#         return np.nan
#     elif ct1 == 0 or np.isnan(ct1):
#         return mean2
#     elif ct2 == 0 or np.isnan(ct2):
#         return mean1
#     else:
#         return (mean1 * ct1 + mean2 * ct2)/(ct1 + ct2)
#
#
# a['wa'] = a[['mean_1', 'ct_1', 'mean_2', 'ct_2']].apply(lambda x: weighted_mean(x['mean_1'], x['ct_1'], x['mean_2'], x['ct_2']), axis=1)
# print(a)
a = pd.DataFrame({'air_store_id': ['a1', 'b2', 'c3', 'd4', 'e5'],
                  'visit_date': ['2017-01-01', '2017-01-02', '2017-01-03', '2017-01-04', '2017-01-05'],
                  'visitor_a': [1, 2, 3, 4, 5]})
print(a)

b = pd.DataFrame({'air_store_id': ['a1', 'b2', 'd4', 'f6', 'g7'],
                  'visit_date': ['2017-01-01', '2017-01-10', '2017-01-03', '2017-01-06', '2017-01-07'],
                  'visitor_b': [1, 2, 3, 6, 7]})
print(b)

c = pd.merge(a, b, on='air_store_id', how='inner', right_index=True)
print(c)
