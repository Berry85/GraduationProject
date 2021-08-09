# -*- coding: utf-8 -*-

import pandas as pd

input_file_path = "./Data/Event.csv"
output_file_path = "./Data/Output.csv"
input_file = pd.read_csv(input_file_path)
output_file = pd.read_csv(output_file_path)

# id,Age,Gender,Place,Object,Measurement_Result,Measurement_Prompt,procedure_occurrence_way,First_condition,TimeStamp,BeginTimeStamp,EndTimeStamp,after_type,before_type,concurrent_type

if __name__ == '__main__':
    input_file = input_file.merge(output_file, on="id", how='left')
    input_file.drop(columns=["id"], inplace=True)
    input_file.to_csv("./model/word2vec.txt", sep=" ", line_terminator="\r", header=None, index=None)
