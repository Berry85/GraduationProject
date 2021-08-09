# -*- coding: utf-8 -*-
import pandas as pd
from datetime import date, datetime
import numpy as np
import math

"""
处理数据集
"""
FILE_PATH = "P://HQU云盘//hqucloudV2//personal_space//毕设-吴怡萱//新的//Data//Event(4).csv"

X_OUTPUT_FILE = "P://HQU云盘//hqucloudV2//personal_space//毕设-吴怡萱//新的//Data//Event.csv"
AFTER_OUTPUT_FILE = "P://HQU云盘//hqucloudV2//personal_space//毕设-吴怡萱//新的//Data//After.csv"
CONCURRENT_OUTPUT_FILE = "P://HQU云盘//hqucloudV2//personal_space//毕设-吴怡萱//新的//Data//Concurrent.csv"
BEFORE_OUTPUT_FILE = "P://HQU云盘//hqucloudV2//personal_space//毕设-吴怡萱//新的//Data//Before.csv"


def age(birth):
    year, month, day = birth.split("-")
    today = date.today()
    Age = int(today.year) - int(year)
    return Age


# 计算BeginTimeStamp和EndTimeStamp间的天数
def period(begin, end):
    time = 0
    if begin == "0000-00-00" or end == "0000-00-00":
        return time
    else:
        begin = datetime.strptime(begin, "%Y-%m-%d")
        end = datetime.strptime(end, "%Y-%m-%d")
        begin = begin.date()
        end = end.date()
        time = end - begin
        time = time.days
        return time


# 计算当时发生的年龄
def condition_age(time, ConditionAge):
    if math.isnan(ConditionAge):
        year, month, day = time.split("-")
        today = date.today()
        ConditionAge = int(today.year) - int(year)
        return int(ConditionAge)
    else:
        return int(ConditionAge)


def concat(df):
    temp = ' '.join(np.unique(df.values))
    return temp


if __name__ == '__main__':
    # 拆分的读取文件
    file_list = []
    for chunk in pd.read_csv(FILE_PATH, encoding='utf-8', sep=',', chunksize=100000):
        file_list.append(chunk)
        print("file_list.shape", len(file_list))
        print("-------------------")
        del chunk
    file = pd.concat(file_list, axis=0, copy=False)
    del file_list
    print("file.shape", file.shape)
    print("-------------------")

    # 删除重复数据
    file.drop_duplicates(inplace=True)
    print("删除重复数据后的数据：")
    print(file.shape)
    print("-------------------")

    # 删除没有id，event_type的数据
    file.dropna(subset=['id'], inplace=True)
    file.dropna(subset=['event_type'], inplace=True)
    print("删除没有id，event_type后的数据：")
    print(file.shape)
    print("-------------------")

    # 删除ConditionAge, FirstCondition, Person_id列
    file.drop(columns=['Person_id'], inplace=True)
    # file.drop(columns=['ConditionAge'], inplace=True)
    file.drop(columns=['First_condition'], inplace=True)
    print("删除累赘列后的数据")
    print(file.shape)
    print("-------------------")

    file['BeginTimeStamp'].fillna("0000-00-00", inplace=True)
    file['EndTimeStamp'].fillna("0000-00-00", inplace=True)

    # 把Birth转成年龄
    file['Age'] = file.apply(lambda x: age(x['Birth']), axis=1)
    # 计算事件发生的年龄
    file['ConditionAge'] = file.apply(lambda x: condition_age(x['Birth'], x['ConditionAge']), axis=1)
    file['DuringTime'] = file.apply(lambda x: period(x['BeginTimeStamp'], x['EndTimeStamp']), axis=1)
    # 删掉Birth
    file.drop(columns=['Birth'], inplace=True)
    print("Birth转成Age后的数据：")
    print(file.shape)
    print("-------------------")

    # 先把nan填充完成
    file['after_type'].fillna("FIN", inplace=True)
    file['before_type'].fillna("FIRST", inplace=True)
    file['concurrent_type'].fillna("Event", inplace=True)

    # 合并数据
    after_type = file.groupby(["id"])['after_type'].apply(concat).reset_index().drop_duplicates()
    print("after_type：")
    print(after_type.shape)
    print("-------------------")

    before_type = file.groupby(["id"])['before_type'].apply(concat).reset_index().drop_duplicates()
    print("before_type：")
    print(before_type.shape)
    print("-------------------")

    concurrent_type = file.groupby(["id"])['concurrent_type'].apply(concat).reset_index().drop_duplicates()
    print("concurrent_type：")
    print(concurrent_type.shape)
    print("-------------------")

    # 删除3列type数据
    file.drop("after_type", axis=1, inplace=True)
    file.drop("before_type", axis=1, inplace=True)
    file.drop("concurrent_type", axis=1, inplace=True)
    file.drop_duplicates(inplace=True)
    print("file：")
    print(file.shape)
    print("-------------------")
    # 拼接
    file = file.merge(after_type, on="id", how='left')
    file = file.merge(before_type, on="id", how='left')
    file = file.merge(concurrent_type, on="id", how='left')
    # 删除时间
    file.drop(columns=['TimeStamp'], inplace=True)
    file.drop(columns=['BeginTimeStamp'], inplace=True)
    file.drop(columns=['EndTimeStamp'], inplace=True)
    # 重新排序
    file = file[["id", "event_type", "Age", "Gender", "Place", "Object", "Measurement_Result",
                 "Measurement_Prompt", "procedure_occurrence_way", "ConditionAge", "DuringTime",
                 "after_type", "before_type", "concurrent_type"]]

    # 把id和file中的数据连接后再拆开
    after_file = file[["id", "after_type"]]
    before_file = file[["id", "before_type"]]
    concurrent_file = file[["id", "concurrent_type"]]

    # 删除3列type数据
    file.drop("after_type", axis=1, inplace=True)
    file.drop("before_type", axis=1, inplace=True)
    file.drop("concurrent_type", axis=1, inplace=True)
    file.drop_duplicates(inplace=True)

    '''
    将
    Place                       的Na转为       科室
    Object                      的Na改成       无
    Measurement_Result          的Na改成       0.0
    Measurement_Prompt          的Na改成       无数据
    procedure_occurrence_way    的Na改成       无麻醉
    '''
    file['Place'].fillna("科室", inplace=True)
    file['Object'].fillna("无", inplace=True)
    file['Measurement_Result'].fillna(0.0, inplace=True)
    file['Measurement_Prompt'].fillna("无数据", inplace=True)
    file['procedure_occurrence_way'].fillna("无麻醉", inplace=True)

    file.to_csv(X_OUTPUT_FILE, sep=",", index=False, line_terminator="\n")

    after_file.to_csv(AFTER_OUTPUT_FILE, sep=" ", index=False, line_terminator="\n")
    before_file.to_csv(BEFORE_OUTPUT_FILE, sep=" ", index=False, line_terminator="\n")
    concurrent_file.to_csv(CONCURRENT_OUTPUT_FILE, sep=" ", index=False, line_terminator="\n")

    print(file.shape)
    print(file)
