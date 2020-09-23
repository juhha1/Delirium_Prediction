import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os, json, pickle

import sys, os

#############################################################################
# get the filename, file directory, etc.
base_dir = os.path.join('/'.join(os.path.abspath(__file__).split('/')[:-1]), '../dataset')

config = {
    'data_fileloc': os.path.join(base_dir, 'raw_data'),
    'save_fileloc': os.path.join(base_dir, 'input_data', 'input_smc.csv'),
    'scaler_fileloc': os.path.join(base_dir, 'input_data', 'scalar_smc.pkl'),
    'diff_hour': 4,
    'cutoff_hour': 48,
    'gcs_cutoff_hour': 24,
    'lab_cutoff_hour': 24 * 7,
    'visit': 'visit9.csv',
    'vent': 'main_vent1.csv',
    'crrt': 'main_crrt1.csv',
    'io': 'main_io1.csv',
    'adl': 'main_ADL_etc1.csv',
    'dementia': 'main_dementia1.csv',
    'medication': 'main_medication2.csv',
    'vital': 'main_vs1.csv',
    'vital_seq': 'main_vs2.csv',
    'lab': 'main_lab2.csv',
    'dx': 'main_dx1.csv',
    'gcs': 'main_gcs1.csv',
    'sofa': 'main_SOFA1.csv',
    'label': 'case_control',
    'visit_cat_cols': ['ward_CD', 'from_NM', 'ICU_in_reason_NM', 'gender'],
    'visit_numeric_cols': ['age', 'LOS_adm_to_wardin']
}

# for datetime
second = 1000000000
minute = second * 60
hour = minute * 60
day = hour * 24

#############################################################################
# Functions
#############################################################################
# visit
def load_visit(config):
    FILELOC = config['data_fileloc']
    VISIT_FNAME = config.get('visit')
    diff_hour = config['diff_hour']
    df_visit = pd.read_csv(os.path.join(FILELOC, VISIT_FNAME))
    df_visit['target_DT'] = pd.to_datetime(df_visit['ward_in_DT']).apply(lambda x: x + timedelta(hours = diff_hour))
    for col in ['ward_in_DT','target_DT','ward_out_DT']:
        df_visit[col] = pd.to_datetime(df_visit[col])
    return df_visit
# Import Visit
def make_data_visit(config):
    # bring visit table
    df_visit = load_visit(config)
    # key cols
    key_cols = ['hadm_ID','ward_in_DT','target_DT']
    # label cols
    label_col1 = ['case_control'] # main label
    label_col2 = ['label'] # optional label
    
    # fix ICU_in_reason_NM - added on 5/4
    new_icu_reason = {
        'Post_CPR': 'Cardiovascular',
        'Post_Intervention': 'Peri_operation',
        'Post_OP': 'Peri_operation',
        'Pre_OP': 'Peri_operation'
    }
    df_visit['ICU_in_reason_NM'] = df_visit['ICU_in_reason_NM'].apply(lambda x: new_icu_reason[x] if new_icu_reason.get(x) else x)
    
    # Get categorical columns: ward_CD, from_NM, ICU_in_reason_NM, gender
    cat_cols = config.get('visit_cat_cols')
    # Get numeric columns: age, LOS_adm_to_wardin
    num_cols = config.get('visit_numeric_cols')
    # only bring necessary columns
    df = df_visit[key_cols + label_col1 + label_col2 + num_cols + cat_cols]
    # categorical data 더미화
    df = pd.get_dummies(df, columns = cat_cols)
    return df
#############################################################################
# ADL
def load_adl(config):
    FILELOC = config['data_fileloc']
    ADL_FNAME = config.get('adl')
    df_adl = pd.read_csv(os.path.join(FILELOC, ADL_FNAME)).iloc[:,1:]
    return df_adl
# Import ADL & merge
def make_data_adl(df, config):
    # bring ADL table
    df_adl = load_adl(config)
    # define categorical columns
    cat_cols = ['sleep_disorder', 'ADL_NM', 'drink_NM']
    # merge
    df = df.merge(df_adl[['hadm_ID'] + cat_cols], on = ['hadm_ID'], how = 'left')
    # categorical data 더미화
    df = pd.get_dummies(df, columns = cat_cols)
    return df
#############################################################################
# DX
def load_dx(config):
    FILELOC = config['data_fileloc']
    DX_FNAME = config.get('dx')
    df_dx = pd.read_csv(os.path.join(FILELOC, DX_FNAME)).iloc[:,1:]
    df_dx.replace(True, 1, inplace = True)
#     df_dx.fillna(0, inplace = True)
    return df_dx
# Import dx and merge
def make_data_dx(df, config):
    # bring DX table (DX table은 벌써 더미화가 되어 있음)
    df_dx = load_dx(config)
    # dummy 화가 되어 있기 때문에 그냥 merge 시키면 됨
    df = df.merge(df_dx, on = ['hadm_ID'], how = 'left')
#     df.fillna(0, inplace = True)
    return df
#############################################################################
# LAB
def load_lab(config):    
    FILELOC = config['data_fileloc']
    LAB_FNAME = config.get('lab')
    df_lab = pd.read_csv(os.path.join(FILELOC, LAB_FNAME)).iloc[:,1:]
    df_lab['report_DT'] = pd.to_datetime(df_lab['report_DT'])
    return df_lab
# import LAB and merge
def make_data_lab(df, config, return_dict = False):
    # bring reference columns
    # hadm_ID for merging
    # ward_in_DT for valid lab 값들 들고올때 (-cutoff_hour ~ lab report_DT ~ t1)
    key_cols = ['hadm_ID','ward_in_DT']
    cutoff_hour = config['lab_cutoff_hour']
    diff_hour = config['diff_hour']
    # bring lab table
    df_lab = load_lab(config)
    numeric_columns = df_lab.columns[df_lab.dtypes == np.float64].tolist()
    lab_columns = ['hadm_ID','report_DT'] + numeric_columns
    df_lab = df_lab[lab_columns]
    # drop cols by lab count
    lab_cols = ['hadm_ID'] + df_lab.columns[3:].tolist()
    temp = df_lab[lab_cols].groupby(['hadm_ID']).agg('max').reset_index()
    lab_value_cols = temp.columns[1:][temp[lab_cols[1:]].apply(lambda x: x.isnull().sum() / temp.shape[0], axis = 0) < 0.6].tolist()
    lab_columns = ['hadm_ID', 'report_DT'] + lab_value_cols
    df_lab = df_lab[lab_columns]
    # 5-7 drop ABGA
    abga_cols = ['BL318508', 'BL318509', ]
    df_lab
    # no outlier - outlier is processed already
    # normalize
    normalize_dict = {} # this is used for external validation - mimic
    list_lab = lab_value_cols
    for col in list_lab:
        mean_ = df_lab[col].mean()
        std_ = df_lab[col].std()
        normalize_dict[col] = {'mean': mean_, 'std': std_}
        df_lab[col] = (df_lab[col] - mean_) / std_
    # min_max_scaler
    min_max_scaler = {} # this is used for external validation - mimic
    for col in list_lab:
        min_ = df_lab[col].min()
        max_ = df_lab[col].max()
        min_max_scaler[col] = {'min': min_, 'max': max_}
        df_lab[col] = df_lab[col].apply(lambda x: (x - min_) / (max_ - min_))
    # bring only valid lab data (check validity by hadm_ID and report_DT)
    # bring only valid hadm_ID
    temp = df[key_cols]
    lab_merge = temp.merge(df_lab, on = ['hadm_ID'], how = 'inner')
    lab_merge['diff_time'] = (lab_merge['report_DT'] - lab_merge['ward_in_DT']).astype(int) / hour
    # bring only (-cutoff_hour ~ report_DT ~ t1)
    cond = lab_merge['diff_time'].apply(lambda x: -cutoff_hour <= x / hour <= diff_hour)    
    lab_merge = lab_merge[cond].drop(columns = ['diff_time'])
    lab_merge = lab_merge.sort_values(['report_DT']) # this is for groupby - keep the latest record
    # merge
    temp = lab_merge[['hadm_ID'] + lab_value_cols]
    temp = temp.groupby(by = ['hadm_ID']).agg(['max','min','mean', 'last']).reset_index()
    temp.columns = ['hadm_ID'] + [i + '_' + j for i,j in temp.columns[1:]]
    df = df.merge(temp, on = ['hadm_ID'], how = 'left')
#     df.fillna(0, inplace = True)
    if return_dict:
        return df, normalize_dict, min_max_scaler
    return df
#############################################################################
# GCS
def load_gcs(config):    
    FILELOC = config['data_fileloc']
    GCS_FNAME = config.get('gcs')
    df_gcs = pd.read_csv(os.path.join(FILELOC, GCS_FNAME)).iloc[:,1:]
    df_gcs['rcrd_DT'] = pd.to_datetime(df_gcs['rcrd_DT'])
    return df_gcs
# Import GCS and merge
def make_data_gcs(df, config, return_dict = False):
    # bring ref columns
    key_cols = ['hadm_ID','ward_in_DT']
    cutoff_hour = config['gcs_cutoff_hour']
    diff_hour = config['diff_hour']
    # bring GCS table
    df_gcs = load_gcs(config)
    # bring valid data by hadm_ID and target_DT
    temp = df[key_cols]
    # bring valid hadm_ID
    gcs_merge = temp.merge(df_gcs, on = ['hadm_ID'], how = 'inner')
    # bring only (-cutoff_hour ~ record_DT ~ t1)
    cond = (gcs_merge['rcrd_DT'] - gcs_merge['ward_in_DT']).astype(int).apply(lambda x: -cutoff_hour <= (x / hour) <= diff_hour)
    gcs_merge = gcs_merge[cond]
    # make E,M,V as a valid type
    gcs_merge['E'] = gcs_merge['E'].astype(float)
    gcs_merge['M'] = gcs_merge['M'].astype(float)
    gcs_merge['V'] = gcs_merge.apply(lambda x: (-0.3756 * (x['M'])) + (0.4233 * (x['E'])) if (str(x['V']) in 'et') else x['V'], axis = 1).astype(float)
    # no normalize_dict and min_max_scaler
    normalize_dict = {}
    min_max_scaler = {}
    # merge
    gcs_merge = gcs_merge.sort_values(['rcrd_DT'])
    gcs_merge = gcs_merge[['hadm_ID', 'E', 'V', 'M']].groupby(['hadm_ID']).agg(['min','max','mean','last']).reset_index()
    gcs_merge.columns = [i if j == '' else i + "@" + j for i,j in gcs_merge.columns]
    df = df.merge(gcs_merge, on = ['hadm_ID'], how = 'left')
    if return_dict:
        return df, normalize_dict, min_max_scaler
    return df   
#############################################################################
# MED
def load_med(config):    
    FILELOC = config['data_fileloc']
    MED_FNAME = config.get('medication')
    df_med = pd.read_csv(os.path.join(FILELOC, MED_FNAME)).iloc[:,1:]
    df_med['ordr_rgst_DT'] = pd.to_datetime(df_med['ordr_rgst_DT'])
    return df_med
# import MED and merge
def make_data_med(df, config):
    # define ref columns
    key_cols = ['hadm_ID','ward_in_DT']
    cutoff_hour = config['cutoff_hour']
    diff_hour = config['diff_hour']
    # bring MED table
    df_med = load_med(config)
    # bring valid data by hadm_ID and record_DT 
    temp = df[key_cols]
    # bring valid hadm_ID
    med_merge = temp.merge(df_med, on = ['hadm_ID'], how = 'inner')
    # bring valid report_DT
    cond = (med_merge['ordr_rgst_DT'] - med_merge['ward_in_DT']).astype(int).apply(lambda x: -cutoff_hour <= (x / hour) <= diff_hour)
    med_merge = med_merge[cond]
    med_merge.sort_values('ordr_rgst_DT', inplace = True)
    med_merge.replace(True, 1, inplace = True)
    med_merge = med_merge.groupby(['hadm_ID'], as_index = False).agg('max')
    # merge
    med_cols = ['hadm_ID'] + med_merge.columns[3:].tolist()
    temp = med_merge[med_cols]
    df = df.merge(temp, on = ['hadm_ID'], how = 'left')
    return df    
#############################################################################
# dementia
def load_dementia(config):
    FILELOC = config['data_fileloc']
    DEMENTIA_FNAME = config.get('dementia')
    df_dementia = pd.read_csv(os.path.join(FILELOC, DEMENTIA_FNAME)).iloc[:,1:]
    df_dementia.replace(True, 1, inplace = True)
    return df_dementia
# import DEMENTIA and merge: 이거는 hadm_ID별 processing이 벌써 되어 있음
def make_data_dementia(df, config):
    df_dementia = load_dementia(config)
    df = df.merge(df_dementia, on = ['hadm_ID'], how = 'left')
#     df.fillna(0, inplace = True)
    return df
#############################################################################
# io
def load_io(config):
    FILELOC = config['data_fileloc']
    IO_FNAME = config.get('io')
    df_io = pd.read_csv(os.path.join(FILELOC, IO_FNAME)).iloc[:,1:]
    df_io['rcrd_DT'] = pd.to_datetime(df_io['rcrd_DT'])
    return df_io
# import IO and merge 
def make_data_io(df, config):
    # define reference columns
    key_cols = ['hadm_ID','ward_in_DT','target_DT']
    cutoff_hour = config['cutoff_hour']
    diff_hour = config['diff_hour']
    # bring IO table
    df_io = load_io(config)
    # bring only valid items by hadm_ID and report_DT
    temp = df[key_cols]
    # bring only valid hadm_ID
    io_merge = temp.merge(df_io, on = ['hadm_ID'], how = 'inner')
    # bring only valid report_DT
    cond = (io_merge['rcrd_DT'] - io_merge['ward_in_DT']).astype(int).apply(lambda x: -cutoff_hour <= x / hour <= diff_hour)
    io_merge = io_merge[cond]
    io_merge.sort_values('rcrd_DT', inplace = True) 
    # merge
    temp = io_merge[['hadm_ID','io_value']].groupby(by = ['hadm_ID']).agg(['mean','max','min','std','last']).reset_index()
    temp.columns = ['hadm_ID'] + [i + '_' + j for i,j in temp.columns[1:]]
    df = df.merge(temp, on = ['hadm_ID'], how = 'left') 
    return df
#############################################################################
# vent
def load_vent(config):
    FILELOC = config['data_fileloc']
    VENT_FNAME = config.get('vent')
    df_vent = pd.read_csv(os.path.join(FILELOC, VENT_FNAME)).iloc[:,1:]
    df_vent['rcrd_DT'] = pd.to_datetime(df_vent['rcrd_DT'])
    df_vent = df_vent[['hadm_ID','rcrd_DT','vent_status']]
    df_vent['vent_status'] = 1
    return df_vent
# import VENT and merge
def make_data_vent(df, config):
    # define reference columns
    key_cols = ['hadm_ID','ward_in_DT']
    diff_hour = config['diff_hour']
    # bring VENT table 
    df_vent = load_vent(config)
    # bring only valid items by hadm_ID and report_DT
    temp = df[key_cols]
    # bring only valid hadm_ID
    vent_merge = temp.merge(df_vent, on = ['hadm_ID'], how = 'inner')
    # vent 기록이 t1 2시간 전에 있으면 1 아니면 0
    vent_merge['record-t1'] = (vent_merge['ward_in_DT'].apply(lambda x: x + timedelta(hours = 4)) - vent_merge['rcrd_DT']).astype(int) / hour
    vent_merge = vent_merge[vent_merge['record-t1'].apply(lambda x: 0 <= x <= 2)]
    vent_merge = vent_merge.drop_duplicates(['hadm_ID'], keep = 'first')
    # merge
    df = df.merge(vent_merge[['hadm_ID','vent_status']], on = 'hadm_ID', how = 'left')
    return df
#############################################################################
# CRRT
def load_crrt(config):
    FILELOC = config['data_fileloc']
    CRRT_FNAME = config.get('crrt')
    df_crrt = pd.read_csv(os.path.join(FILELOC, CRRT_FNAME)).iloc[:,1:]
    df_crrt['rcrd_DT'] = pd.to_datetime(df_crrt['rcrd_DT'])
    df_crrt = df_crrt[['hadm_ID','rcrd_DT','crrt_status']]
    df_crrt['crrt_status'] = 1
    return df_crrt
# import CRRT and merge
def make_data_crrt(df, config):
    # reference cols
    key_cols = ['hadm_ID','ward_in_DT']
    diff_hour = config['diff_hour']
    # bring crrt table
    df_crrt = load_crrt(config)
    # bring only valid data by hadm_ID and record_DT
    temp = df[key_cols]
    # bring valid hadm_ID
    crrt_merge = temp.merge(df_crrt, on = ['hadm_ID'], how = 'inner')
    # crrt 기록이 t1 2시간 전에 있으면 1 아니면 0
    crrt_merge['record-t1'] = (crrt_merge['ward_in_DT'].apply(lambda x: x + timedelta(hours = 4)) - crrt_merge['rcrd_DT']).astype(int) / hour
    crrt_merge = crrt_merge[crrt_merge['record-t1'].apply(lambda x: 0 <= x <= 2)]
    crrt_merge = crrt_merge.drop_duplicates(['hadm_ID'], keep = 'first')
    # merge
    df = df.merge(crrt_merge[['hadm_ID','crrt_status']], on = 'hadm_ID', how = 'left')
    return df
#############################################################################
# vital no seq
def load_vital(config):
    FILELOC = config['data_fileloc']
    VITAL_FNAME = config.get('vital')
    df_vs = pd.read_csv(os.path.join(FILELOC, VITAL_FNAME)).iloc[:,1:]
    df_vs['rcrd_DT'] = pd.to_datetime(df_vs['rcrd_DT'])
    return df_vs
# import VITAL and merge
def make_data_vital(df, config, return_dict = False):
    # define reference cols
    key_cols = ['hadm_ID','ward_in_DT','target_DT']
    cutoff_hour = config['cutoff_hour']
    diff_hour = config['diff_hour']
    df_vs = load_vital(config)
    # outlier 제거는 벌써 되었음 - pre-processing-2 참조 
    outlier_dict = {}
    # normalize
    normalize_dict = {}
    list_vital = df_vs.columns[2:]
    for col in list_vital:
        mean_ = df_vs[col].mean()
        std_ = df_vs[col].std()
        normalize_dict[col] = {'mean': mean_, 'std': std_}
        df_vs[col] = (df_vs[col] - mean_) / std_
    # min_max_scaler
    min_max_scaler = {}
    for col in list_vital:
        min_ = df_vs[col].min()
        max_ = df_vs[col].max()
        min_max_scaler[col] = {'min': min_, 'max': max_}
        df_vs[col] = df_vs[col].apply(lambda x: (x - min_) / (max_ - min_))
    # bring only valid data - by hadm_ID and record_DT
    temp = df[key_cols]
    # bring only valid hadm_ID
    vital_merge = temp.merge(df_vs, on = ['hadm_ID'], how = 'inner')
    # bring only valid record_DT
    cond = (vital_merge['rcrd_DT'].astype(int) - vital_merge['ward_in_DT'].astype(int)).apply(lambda x: -cutoff_hour <= x / hour <= diff_hour)
    vital_merge = vital_merge[cond]    
    # merge
    vital_cols = ['hadm_ID', 'SBP', 'DBP', 'HR', 'RR', 'BT', 'SpO2']
    temp = vital_merge[vital_cols].groupby(by = ['hadm_ID']).agg(['mean','max','min','std', 'last']).reset_index()
    temp.columns = ['hadm_ID'] + [i + '_' + j for i,j in temp.columns[1:]]
    df = df.merge(temp, on = ['hadm_ID'], how = 'left')
    if return_dict:
        return df, outlier_dict, normalize_dict, min_max_scaler
    return df
#############################################################################
# SOFA
def load_sofa(config):
    FILELOC = config['data_fileloc']
    SOFA_FNAME = config.get('sofa')
    df_sofa = pd.read_csv(os.path.join(FILELOC, SOFA_FNAME)).iloc[:,1:]
    return df_sofa
# import SOFA and merge
def make_data_sofa(df, config, return_dict = False):
    # load sofa
    df_sofa = load_sofa(config)
    # normalizer
    normalize_dict = {}
    mean_ = df_sofa['SOFA_score'].mean()
    std_ = df_sofa['SOFA_score'].std()
    normalize_dict['SOFA_score'] = {'mean': mean_, 'std': std_}
    df_sofa['SOFA_score'] = (df_sofa['SOFA_score'] - mean_) / std_
    # min_max_scaler
    min_max_scaler = {}
    min_ = df_sofa['SOFA_score'].min()
    max_ = df_sofa['SOFA_score'].max()
    min_max_scaler['SOFA_score'] = {'min': min_, 'max': max_}
    df_sofa['SOFA_score'] = (df_sofa['SOFA_score'] - min_) / (max_ - min_)
    # merge
    df = df.merge(df_sofa, on = ['hadm_ID'], how = 'left')
#     df.fillna(0, inplace = True)
    if return_dict:
        return df, normalize_dict, min_max_scaler
    return df

def main():
    #############################################################################
    # Generate data
    df = make_data_visit(config)
    print('after merging visit', df.shape)
    df = make_data_adl(df, config)
    print('after merging adl', df.shape)
    df = make_data_dx(df, config)
    print('after merging dx', df.shape)
    df, lab_norm_dict, lab_scaler_dict = make_data_lab(df, config, return_dict = True)
    print('after merging lab', df.shape)
    df, gcs_norm_dict, gcs_scaler_dict = make_data_gcs(df, config, return_dict = True)
    print('after merging gcs', df.shape)
    df = make_data_med(df, config)
    print('after merging med', df.shape)
    df = make_data_dementia(df, config)
    print('after merging dementia', df.shape)
    df = make_data_io(df, config)
    print('after merging io', df.shape)
    df = make_data_vent(df, config)
    print('after merging vent', df.shape)
    df, vital_out_dict, vital_norm_dict, vital_scaler_dict = make_data_vital(df, config, return_dict = True)
    print('after merging vital', df.shape)
    df = make_data_crrt(df, config)
    print('after merging crrt', df.shape)
    df, sofa_norm_dict, sofa_scaler_dict = make_data_sofa(df, config, return_dict = True)
    print('after merging sofa', df.shape)

    df.to_csv(config['save_fileloc'])
    print('Input file saved in {}'.format(config['save_fileloc']))

    numeric_process_dict = {
        'lab': {
            'norm': lab_norm_dict,
            'min_max_scaler': lab_scaler_dict,
        },
        'gcs':{
            'norm': gcs_norm_dict,
            'min_max_scaler': gcs_scaler_dict
        },
        'vital':{
            'outlier': vital_out_dict,
            'norm': vital_norm_dict,
            'min_max_scaler': vital_scaler_dict
        },
        'sofa':{
            'norm': sofa_norm_dict,
            'min_max_scaler': sofa_scaler_dict
        }
    }
    pickle.dump(numeric_process_dict, open(config['scaler_fileloc'], 'wb'))
    print('Scaler file saved in {}'.format(config['scaler_fileloc']))

if __name__ == '__main__':
    main()