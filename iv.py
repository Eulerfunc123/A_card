#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:25:14 2019

@author: liuan
"""
from datetime import date, timedelta
import pandas as pd
import numpy as np


class ThirdDatasource(object):
    
    def __init__(self, drop_list = None):
        self.drop_dict = {'duplicate_value':[], 'null':[]}
        self.yes_date = str((date.today() + timedelta(days = -1)).strftime("%Y%m%d"))
    

    def calc_score_median(self, sample_set, var):
        var_list = list(np.unique(sample_set[var]))
        var_median_list = []
        for i in range(len(var_list) -1):
            var_median = (var_list[i] + var_list[i+1]) / 2
            var_median_list.append(var_median)
        return var_median_list
    
    
    def choose_best_split(self, sample_set, var, y_var, min_sample):
        score_median_list = self.calc_score_median(sample_set, var)
        median_len = len(score_median_list)
        sample_cnt = sample_set.shape[0]
        sample1_cnt = sum(sample_set[y_var])
        sample0_cnt =  sample_cnt- sample1_cnt
        Gini = 1 - np.square(sample1_cnt / sample_cnt) - np.square(sample0_cnt / sample_cnt)
        bestGini = 0.0; bestSplit_point = 0.0; bestSplit_position = 0.0
        for i in range(median_len):
            left = sample_set[sample_set[var] < score_median_list[i]]
            right = sample_set[sample_set[var] > score_median_list[i]]
            
            left_cnt = left.shape[0]; right_cnt = right.shape[0]
            left1_cnt = sum(left[y_var])
            right1_cnt = sum(right[y_var])
            left0_cnt =  left_cnt - left1_cnt
            right0_cnt =  right_cnt - right1_cnt
            left_ratio = left_cnt / sample_cnt
            right_ratio = right_cnt / sample_cnt
            
            if left_cnt < min_sample or right_cnt < min_sample:
                continue
            Gini_left = 1 - np.square(left1_cnt / left_cnt) - np.square(left0_cnt / left_cnt)
            Gini_right = 1 - np.square(right1_cnt / right_cnt) - np.square(right0_cnt / right_cnt)
            Gini_temp = Gini - (left_ratio * Gini_left + right_ratio * Gini_right)
            if Gini_temp > bestGini:
                bestGini = Gini_temp; bestSplit_point = score_median_list[i]
                if median_len > 1:
                    bestSplit_position = i / (median_len - 1)
                else:
                    bestSplit_position = i / median_len
            else:
                continue
                   
        Gini = Gini - bestGini
        return bestSplit_point, bestSplit_position
    
    
    def bining_data_split(self, sample_set, var, y_var, min_sample, split_list):
        split, position = self.choose_best_split(sample_set, var, y_var, min_sample)
        if split != 0.0:
            split_list.append(split)
        sample_set_left = sample_set[sample_set[var] < split]
        sample_set_right = sample_set[sample_set[var] > split]
        if len(sample_set_left) >= min_sample * 2 and position not in [0.0, 1.0]:
            self.bining_data_split(sample_set_left, var, y_var, min_sample, split_list)
        else:
            None
        if len(sample_set_right) >= min_sample * 2 and position not in [0.0, 1.0]:
            self.bining_data_split(sample_set_right, var, y_var, min_sample, split_list)
        else:
            None
    
    
    def get_bestsplit_list(self, sample_set, var, y_var):
        min_df = sample_set.shape[0] * 0.1
        split_list = []
        self.bining_data_split(sample_set, var, y_var, min_df, split_list)
        split_list.append(max(sample_set[var])+1)
        split_list.append(0)
        split_list.append(min(sample_set[var])-1)
        return sorted(split_list)
        
    
    # 保存特征描述统计结果
    def descriptive_statistics(self, dataframe, storage_path):
        res = pd.DataFrame(index = ['非缺失值数量', '缺失值数量', '平均值',
                                    '标准差', '最小值', '25%分位数', '中位数',
                                    '75%分位数', '最大值'])
        for col in dataframe.columns:
            try:
                first_value = [i for i in dataframe[col] if i != -1][0]
                if isinstance(first_value, np.int64) or isinstance(first_value, np.float64) \
                or isinstance(first_value, int) or isinstance(first_value, float):
                    tmp = list(dataframe[col].describe())                
                    tmp.insert(1, len(dataframe[col][pd.isnull(dataframe[col])]))
                    res[col] = tmp
            except Exception:
                self.drop_dict['null'].append(col)
        res.to_csv(storage_path+'feature_describe{}.csv'.format(self.yes_date), encoding = 'utf-8', index = True)
 
       
    # 对特征进行分箱   
    def split_box(self, x, y, dup_per_low, null_per_low):
        dic = {x.name:x, 'target':y}
        x_y_df = pd.DataFrame(dic)
        # 若变量某个值比例超过dup_per_low，视为过多重复值
        dup_per = x_y_df[x.name].value_counts().max()/float(len(x_y_df))
        if dup_per >= dup_per_low:
            print("The varible {} may not be useful because of too many duplicate value".format(x.name))
            self.drop_dict['duplicate_value'].append(x.name)
        else:
            # 若变量的缺失值比例超过null_per_low，视为过多缺失值
            null_per = len(x_y_df[x.name][pd.isnull(x_y_df[x.name])])
            if null_per >= null_per_low:
                print("The varible {} may not be useful because of too many null value".format(x.name))
                self.drop_dict['null'].append(x.name)
            else:
                # 符合条件，进行分箱
                bins = self.get_bestsplit_list(x_y_df, x.name, 'target')
                after_cut = pd.cut(list(x_y_df[x.name]), bins, right = True)
                after_cut = after_cut.codes
        return pd.DataFrame({x.name:after_cut, 'target':y})
                
               
    # 计算特征IV值
    def calIV(self, x, y):
        dic = {x.name:x, y.name:y}
        x_y_df = pd.DataFrame(dic)
        col = x.name
        target = y.name
        x_sum = pd.DataFrame(x_y_df.groupby(col)[col].count())
        y_sum = pd.DataFrame(x_y_df.groupby(col)[target].sum())
        x_y_sum = pd.DataFrame(pd.merge(x_sum, y_sum, how = "left", left_index = True, right_index = True))
        b_total = x_y_sum[y.name].sum()
        total = x_y_sum[x.name].sum()
        g_total = total - b_total
        x_y_sum["bad"] = x_y_sum.apply(lambda x: x[col] / b_total, axis = 1)
        x_y_sum["good"] = x_y_sum.apply(lambda x: (x[col] - x[target]) / g_total, axis = 1)
        x_y_sum["WOE"] = x_y_sum.apply(lambda x: np.log(x.bad / x.good),axis = 1)
        tmp = x_y_sum.loc[:,["bad", "good", "WOE"]]
        tmp["IV"] = tmp.apply(lambda x:(x.bad - x.good) * x.WOE, axis = 1)
        IV = sum(tmp["IV"])
        return IV
    

if __name__ == '__main__':
    # 测试数据
    test_data = pd.read_csv('/users/eulerfunction/desktop/test_data_iv.csv')
    tds = ThirdDatasource()
    # 保存描述统计结果
    tds.descriptive_statistics(test_data, '/users/eulerfunction/desktop/')
    # 数值型变量分箱后计算IV
    after_split_box = tds.split_box(test_data['final_score'], test_data['od_days_class'], 0.95, 0.7)
    print(tds.calIV(after_split_box['final_score'], after_split_box['target']))
    # 分类型变量计算IV
    print(tds.calIV(test_data['final_decision'], test_data['od_days_class'])) 
