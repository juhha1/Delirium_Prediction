import numpy as np
import pandas as pd
import psycopg2, os, pickle
import json
from datetime import datetime, timedelta

#############################################################################
# Configuration
# list config
diff_hour = 4
cutoff_hour = 48
gcs_cutoff_hour = 24
lab_cutoff_hour = 24 * 7
base_dir = os.path.join('/'.join(os.path.abspath(__file__).split('/')[:-1]), '../dataset')
scaler_fileloc = os.path.join(base_dir, 'input_data', 'scalar_smc.pkl')
save_fileloc = os.path.join(base_dir, 'input_data', 'input_mimic.csv')
outlier_dict_vital = {
    'SBP': {'min': 0, 'max': 300},
    'DBP': {'min': 0, 'max': 200},
    'HR': {'min': 0, 'max': 300},
    'SpO2': {'min': 0, 'max': 100}
}

# for datetime
second = 1000000000
minute = second * 60
hour = minute * 60
day = hour * 24

#############################################################################
# Mimic connection to SQL
# sql connection
#################################
# fill this out to run code
#################################
sqluser = ''
dbname = ''
schema_name = ''
sqlpassword = ''

query_schema = 'set search_path to ' + schema_name + ';'
con = psycopg2.connect(dbname = dbname, user = sqluser, password = sqlpassword)


#############################################################################
# Make tables
"""
1. Bring study population
    a. bring cam_icu itemid
    b. bring hadm_id for study population
    c. bring study population by rule - selection diagram
        c-1. get icu_stay population
        c-2. rule 1: age > 18
        c-3. rule 2: have at least one record between 0-4 AND 4-24
        c-4. rule 3: all icu_records N between 0-4
2. Bring tables (with some pre-processing)
3. Pre-processing
    a. get scaler created from shl
    b. scaling - lab, vital
4. Merge data

"""
#############################################################################
######
# 1. Bring study population
print('1. Bring study population...')
# a. bring cam_icu itemid
query = query_schema + """
SELECT *
FROM d_items
WHERE lower(label) like ('%delirium%')
"""
d_items = pd.read_sql_query(query, con)
cam_icu_itemid_list = d_items['itemid'].tolist()
cam_icu_itemid_str = ','.join([str(i) for i in d_items['itemid'].tolist()])

######
# b. bring hadm_id for study population
# bring hadm_id from chartevents
query = query_schema + """
SELECT *
FROM chartevents
WHERE itemid IN ({})
""".format(cam_icu_itemid_str)
chartevents_cam = pd.read_sql_query(query, con)
chartevents_cam['value']=chartevents_cam['value'].astype(str).apply(lambda x: x.lower().split()[0])
hadm_id_set = chartevents_cam['hadm_id'].unique()
hadm_id_set_str = ','.join(["'" + str(i) + "'" for i in hadm_id_set])

######
# uta -> positive
print(chartevents_cam['value'].value_counts())
chartevents_cam['value'] = chartevents_cam['value'].apply(lambda x: 'negative' if x == 'negative' else 'positive')
# after changing
chartevents_cam['value'].value_counts()

# c. Bring study population by rule - based on selection diagram
# bring transfers
query = query_schema + """
    SELECT *
    FROM transfers
    WHERE hadm_id IN ({})
""".format(hadm_id_set_str)
transfers = pd.read_sql_query(query, con)
transfers['new_hadm_id'] = transfers['hadm_id'].astype(str) + '@' + transfers['intime'].astype(str)

print('Original transfers: {}'.format(transfers.shape[0]))
# rule 1- 중환자실 stay
transfers = transfers[transfers['curr_careunit'].isnull() == False]
print('ICU stay population: {}'.format(transfers.shape[0]))

# rule 2 - 나이 18세 이상
# 먼저 patients table 가져와야함 - age
pt_subject_id_str = ','.join(['\'{}\''.format(i) for i in transfers['subject_id'].unique()])
query = query_schema + """
    SELECT *
    FROM patients
    WHERE subject_id IN ({})
""".format(pt_subject_id_str)
patients = pd.read_sql_query(query, con)
temp_cols1 = ['subject_id', 'hadm_id', 'icustay_id', 'dbsource',
       'eventtype', 'prev_careunit', 'curr_careunit', 'prev_wardid',
       'curr_wardid', 'intime', 'outtime', 'los', 'new_hadm_id']
temp_cols2 = ['subject_id', 'gender', 'dob', 'dod', 'dod_hosp', 'dod_ssn',
       'expire_flag']
transfers = transfers[temp_cols1].merge(patients[temp_cols2], on = ['subject_id'], how = 'inner')
transfers['age'] = transfers[['intime', 'dob']].apply(lambda x: x[0].year - x[1].year - ((x[0].month, x[0].day) < (x[1].month, x[1].day)), axis = 1)
transfers = transfers[transfers['age'] >= 18]
print('After rule 1 (age): {}'.format(transfers.shape[0]))
hadm_id_set = transfers['hadm_id'].unique().tolist()
hadm_id_set_str = ','.join(str(i) for i in hadm_id_set)

# rule 3
temp = transfers[['hadm_id', 'new_hadm_id', 'subject_id', 'intime', 'outtime','age']].merge(chartevents_cam[['hadm_id', 'charttime', 'value']], on = ['hadm_id'], how = 'inner')
temp = temp[temp['outtime'] >= temp['charttime']]
temp['time_took'] = (temp['charttime'] - temp['intime']).astype(int).apply(lambda x: x / hour)

# rule 3.1 - 0-4 시간 동안 기록이 하나라도 있는 사람
temp['t1'] = temp['time_took'].apply(lambda x: 0 <= x <= 4)
# rule 3.2 - 4~24시간 동안 기록이 하나라도 있는 사람
temp['t2'] = temp['time_took'].apply(lambda x: 4 < x <= 24)
hadm_id_set_rule3 = set(temp[temp['t1']]['new_hadm_id'].unique().tolist()).intersection(set(temp[temp['t2']]['new_hadm_id'].unique().tolist()))
print('After rule 2 (has data (t0-t1) AND (t1-t2)): {}'.format(len(hadm_id_set_rule3)))

# rule 4 - 처음 4시간동안 모두 N인사람
rule4 = temp[temp['t1']][['new_hadm_id', 'value']].groupby(['new_hadm_id']).apply(lambda x: 'positive' not in set(x['value']))
hadm_id_set_rule4 = set(rule4[rule4].index.tolist())
hadm_id_set_all = hadm_id_set_rule3.intersection(hadm_id_set_rule4)
print('After rule 3 (all Ns during t0-t1): {}'.format(len(hadm_id_set_all)))

# case and control
temp_t2 = temp[temp['t2']]
temp_case = temp_t2[temp_t2['new_hadm_id'].apply(lambda x: x in hadm_id_set_all)].groupby(['new_hadm_id']).apply(lambda x: 'positive' in set(x['value']))
hadm_id_t1_no_t2_no = set(temp_case[temp_case == False].index)
hadm_id_t1_no_t2_yes = set(temp_case[temp_case].index)
subject_id_set_all = set(temp[temp['new_hadm_id'].apply(lambda x: x in hadm_id_set_all)]['subject_id'].unique().tolist())
print('Study population:')
print('all: {}\ncase: {}\ncontrol: {}'.format(len(hadm_id_set_all), len(hadm_id_t1_no_t2_yes), len(hadm_id_t1_no_t2_no)))
print('subject_id: {}'.format(len(subject_id_set_all)))

subject_id_all_str = ','.join([str(i) for i in list(subject_id_set_all)])
hadm_id_all_raw_str = ','.join([str(i) for i in list(temp['hadm_id'].unique().tolist())])
new_hadm_id_case = hadm_id_t1_no_t2_yes
new_hadm_id_control = hadm_id_t1_no_t2_no

#############################################################################
# 2. Bring tables (with some pre-processing)
######
# Transfers table
query = query_schema + """
    SELECT *
    FROM transfers
    WHERE hadm_id IN ({})
""".format(hadm_id_all_raw_str)
transfers = pd.read_sql_query(query, con)

transfers['new_hadm_id'] = transfers['hadm_id'].astype(str) + '@' + transfers['intime'].astype(str)
transfers = transfers[transfers['new_hadm_id'].apply(lambda x: x in hadm_id_set_all)]

# case_control
transfers['case_control'] = transfers['new_hadm_id'].apply(lambda x: 'case' if x in new_hadm_id_case else 'control')
# map to SHL format
map_ward_in = {
    'MICU': 'Internal Medicine Group',
    'CCU': 'Internal Medicine Group',
    'TSICU': 'Surgical Medicine Group',
    'SICU': 'Surgical Medicine Group',
    'CSRU': 'Surgical Medicine Group'
}
# bring transfers
transfers['ward_CD'] = transfers['curr_careunit'].apply(lambda x: map_ward_in.get(x))

######
# Patients table
query = query_schema + """
    SELECT *
    FROM patients
    WHERE subject_id IN ({})
""".format(subject_id_all_str)
patients = pd.read_sql_query(query, con)

######
# Services table - 이거는 이제 필요 없음 (5/7 이후)
query = query_schema + """
    SELECT *
    FROM services
    WHERE hadm_id IN ({})
""".format(hadm_id_all_raw_str)
services = pd.read_sql_query(query, con)

# define mapping items to SHL format
map_reason = {
    'CMED': 'Cardiovascular',
    'VSURG': 'Cardiovascular',
    'SCURU': 'Post_OP',
    'SURG': 'Post_OP',
    'TSURG': 'Post_OP'
}

# make new_hadm_id for merging reference
services['new_hadm_id'] = services['hadm_id'].astype(str) + '@' + services['transfertime'].astype(str)
# mapping to SHL format
services['ICU_in_reason_NM'] = services['curr_service'].apply(lambda x: map_reason.get(x))

######
# Chartevents table
######
# Vital
# mimic_itemids for target vital
vital_itemid = {
    'SBP': [51, 455, 220179, 220050, 225309],
    'DBP': [8368, 8441, 220180, 220051, 225310],
    'SpO2': [646, 220277],
    'HR': [211, 220045]
}

# 데이터 불러올때 쓸 ref itemids
vital_itemid_list = [j for i in list(vital_itemid.values()) for j in i]
vital_itemid_list_str = ','.join([str(i) for i in vital_itemid_list])

# bring vital table by hadm_id & itemid
query = query_schema + """
    SELECT *
    FROM chartevents
    WHERE hadm_id IN ({}) AND itemid IN ({})
""".format(hadm_id_all_raw_str, vital_itemid_list_str)
vital = pd.read_sql_query(query, con)

vital_itemid_map = {i_:j for i,j in zip(vital_itemid.values(), vital_itemid.keys()) for i_ in i}
# vital_type으로 map
vital['vital_type'] = vital['itemid'].apply(lambda x: vital_itemid_map[x])
# 필요한 컬럼들만 들고오기 
vital = vital[['hadm_id', 'vital_type', 'valuenum', 'charttime']]

######
# Sleep disorder
# insomnia 를 key-word로 sleep_disorder에 관한 itemid 들고오기
query = query_schema + """
    SELECT *
    FROM d_items
    WHERE lower(label) like '%insomnia%'
"""
d_items = pd.read_sql_query(query, con)
list_itemids = ','.join([str(i) for i in d_items['itemid'].values.tolist()])

# bring sleep table by hadm_id & itemid
query = query_schema + """
    SELECT *
    FROM chartevents
    WHERE hadm_id IN ({}) AND itemid IN ({})
""".format(hadm_id_all_raw_str, list_itemids)
sleep = pd.read_sql_query(query, con)

######
# ADL
# ADL itemid
adl_itemid = 225092 # Self ADL
# bring adl table by hadm_id & itemid
query = query_schema + """
    SELECT *
    FROM chartevents
    WHERE hadm_id IN ({}) AND itemid IN ({})
""".format(hadm_id_all_raw_str, adl_itemid)
adl = pd.read_sql_query(query, con)

######
# drink
# drink에 대한 itemid 가져오기: by key-word 'etoh'
query = query_schema + """
    SELECT *
    FROM d_items
    WHERE lower(label) like '%etoh%'
"""
d_item = pd.read_sql_query(query, con)
itemid_str = ','.join([str(i) for i in d_item['itemid'].tolist()])
# bring drink table by hadm_id & itemid
query = query_schema + """
    SELECT *
    FROM chartevents
    WHERE hadm_id IN ({}) AND itemid IN ({})
""".format(hadm_id_all_raw_str, itemid_str)
drink = pd.read_sql_query(query, con)

######
# crrt
# crrt에 대한 itemid 가져오기: by key-word 'crrt'
# 227290 (crrt-mode)밖에 없음
query = query_schema + """
    SELECT *
    FROM d_items
    WHERE lower(label) like '%crrt%'
"""
d_item = pd.read_sql_query(query, con)
itemid_str = ','.join([str(i) for i in d_item['itemid'].tolist()])
# bring crrt table
query = query_schema + """
    SELECT *
    FROM chartevents
    WHERE hadm_id IN ({}) AND itemid IN (227290)
""".format(hadm_id_all_raw_str)
crrt = pd.read_sql_query(query, con)

######
# GCS
# gcs에 대한 itemid 가져오기: by key-word 'gcs'
# 220739 (E), 223900 (V), 223901 (M) 밖에 없음
query = query_schema + """
    SELECT *
    FROM d_items
    WHERE lower(label) like '%gcs%'
"""
d_item = pd.read_sql_query(query, con)
itemid_str = ','.join([str(i) for i in d_item['itemid'].tolist()])
# gcs_type map
map_gcs = {
    220739: 'E',
    223900: 'V',
    223901: 'M'
}
# bring gcs table by hadm_id & itemid
query = query_schema + """
    SELECT *
    FROM chartevents
    WHERE hadm_id IN ({}) AND itemid IN ({})
""".format(hadm_id_all_raw_str, itemid_str)
gcs = pd.read_sql_query(query, con)
gcs['itemid'] = gcs['itemid'].apply(lambda x: map_gcs.get(x))

######
# Lab table
# define: lab_name (SHL) & unit & itemid @ lab_code (SHL)
temp = """
WBC Count, Blood: K/uL@51300@BL2011
WBC Count, Blood: K/uL@51301@BL2011
Hematocrit, Blood: %@51221@BL2014
Platelet Count, Blood: K/uL@51265@BL2016
PT(INR): None@51237@BL211103
APTT: sec@51275@BL2112
Protein, Total: g/dL@50976@BL3111
Albumin: g/dL@50862@BL3112
Bilirubin, Total: mg/dL@50885@BL3114
AST: IU/L@50878@BL3115
ALT: IU/L@50861@BL3116
Glucose, Fasting: mg/dL@50809@BL3118
Glucose, Fasting: mg/dL@50931@BL3118
Glucose: mg/dL@50809@BL318512
Glucose: mg/dL@50931@BL318512
BUN: mg/dL@51006@BL3119
Creatinine: mg/dL@50912@BL3120
Na (ABGA): mEq/L@50824@BL318508
Na: mEq/L@83@BL3131
Potassium (K): mEq/L@50971@BL3132
CRP, Quantitative (High Sensitivity): MG/DL@50889@BL3140
CRP, Quantitative (High Sensitivity): mg/L@50889@BL3140
CRP, Quantitative (High Sensitivity): mg/dL@50889@BL3140
Lactic Acid: mmol/L@50813@BL3157
Lactic Acid, Whole blood: mmol/L@50813@BL319301
pH: UNITS@50820@BL318501
pH: units@50820@BL318501
pCO2: MM HG@50818@BL318502
pCO2: mm Hg@50818@BL318502
pO2: MM HG@50821@BL318503
pO2: mm Hg@50821@BL318503
HCO3: mEq/L@50803@BL318504
HCO3: mEq/L@50882@BL318504
O2 Saturation: %@50817@BL318507
O2 Saturation: None@50817@BL318507
K (ABGA): mEq/L@50822@BL318509
"""
our_lab_name = [i.split(':')[0] for i in temp.split('\n')[1:-1]]
mimic_lab_unit = [i.split(':')[1].split('@')[0].strip() for i in temp.split('\n')[1:-1]]
mimic_lab_itemid = [i.split(':')[1].split('@')[1] for i in temp.split('\n')[1:-1]]
mimic_lab_code = [i.split(':')[1].split('@')[2] for i in temp.split('\n')[1:-1]]
mimic_lab_itemid_str = ','.join(mimic_lab_itemid)
lab_itemid_dict = {}
for a,b,c,d in zip(mimic_lab_itemid, our_lab_name, mimic_lab_unit, mimic_lab_code):
    lab_itemid_dict[a] = {
        'our_labname': b,
        'unit': c,
        'our_labcode': d
    }
    
# bring lab table by hadm_id & itemid
query = query_schema + """
    SELECT *
    FROM labevents
    WHERE hadm_id IN ({}) AND itemid IN ({})
""".format(hadm_id_all_raw_str, mimic_lab_itemid_str)
lab = pd.read_sql_query(query, con)

# map to lab_name (SHL)
lab['lab_name'] = lab['itemid'].apply(lambda x: lab_itemid_dict[str(x)]['our_labname'])
# map to lab_code (SHL)
lab['lab_code'] = lab['itemid'].apply(lambda x: lab_itemid_dict[str(x)]['our_labcode'])
# unit수 다른거 고쳐주기 (itemid==50889)
lab['valuenum'] = lab[['itemid', 'valuenum']].apply(lambda x: x[1] / 10 if x[0] == 50889 else x[1], axis = 1)

######
# Prescriptions table
# drug_name (SHL pre-processing 참조)
drug_antibio = ['Azithromycin', 'CeftriaXONE', 'Vancomycin', 'Piperacillin-Tazobactam','Ciprofloxacin IV','Meropenem','CefePIME','Ampicillin-Sulbactam','Aztreonam','Levofloxacin','CefazoLIN','CefTAZidime','Erythromycin', 'AcetaZOLamide Sodium','Acyclovir','Ampicillin Sodium','*NF* Ertapenem Sodium','Linezolid','Daptomycin','Penicillin G Potassium','Nafcillin','Gentamicin','Gentamicin Sulfate','Micafungin','Ambisome','Tigecycline','Tobramycin Sulfate','Bleomycin','DOXOrubicin','Cyclophosphamide','Penicillin G K Desensitization','Ceftaroline','Voriconazole','Sulfamethoxazole-Trimethoprim','Amikacin','Foscarnet Sodium','Ganciclovir','Meropenem Desensitization','Tobramycin']
drug_anticho = ['ChlorproMAZINE']
drug_benzo = ['Midazolam']
drug_inotropics = ['NORepinephrine','DOPamine','EPINEPHrine','Vasopressin','DOBUTamine','Epinephrine','Norepinephrine','Norepinephrine Bitartrate']
drug_mis = ['Propofol', 'Dexmedetomidine', 'Ketamine (For Intubation)','Ketamine']
drug_opiate = ['Fentanyl Citrate','HYDROmorphone (Dilaudid)','Morphine Sulfate','HYDROmorphone','HYDROmorphone-HP','Morphine Infusion ? Comfort Care Guidelines']
drug_steroid = ['Dexamethasone','Hydrocortisone Na Succ.','MethylPREDNISolone Sodium Succ','Dexamethasone Sod Phosphate','Hydrocortisone Study Drug (*IND*)','Hydrocortisone Na Succ']

drug_str = ','.join(['\'{}\''.format(i) for i in drug_antibio + drug_anticho + drug_benzo + drug_inotropics + drug_mis + drug_opiate + drug_steroid])
shl_drug_list = ['Antibiotics', 'anticholinergic_antipsychotics', 'benzodiazepines', 'Inotropics', 'Miscellaneous Antidepressants', 'miscellaneous_antidepressants_anxiolytics_sedatives_hypnotics', 'Opiate Agonists', 'steroids']

drug_mapping = {c:a for a,b in zip(shl_drug_list, [drug_antibio, drug_anticho, drug_benzo, drug_inotropics, drug_mis, drug_opiate, drug_steroid]) for c in b}

# bring drug table by hadm_id & drug_name
query = query_schema + """
    SELECT *
    FROM prescriptions
    WHERE hadm_id IN ({}) AND drug IN ({})
""".format(hadm_id_all_raw_str, drug_str)
drug = pd.read_sql_query(query, con)
# map to drug_name (SHL)
drug['drug'] = drug['drug'].apply(lambda x: drug_mapping[x])
drug = drug[['hadm_id', 'startdate', 'enddate', 'drug']]

######
# diagnoses_icd
# diag_type (SHL)
list_1 = ['MI', 'CHF', 'PVD', 'Stroke', 'Dementia', 'Pulmonary', 'Rheumatic', 'PUD', 'LiverMild', 'DM']
list_2 = ['Cancer', 'Paralysis', 'Renal', 'DMcx']
list_3 = ['LiverSevers']
list_4 = ['HIV', 'Mets']

# helper functions to calculate CCI
def calc_func(x_dict, list_cci, score):
    result_score = 0
    for i in list_cci:
        if x_dict.get(i) == 1:
            result_score += score
    return result_score

def get_age_score(x):
    if 50 <= x < 60:
        return 1
    elif 60 <= x < 70:
        return 2
    elif 70 <= x < 80:
        return 3
    elif 80 <= x:
        return 4
    else:
        return 0

def cci_calculation_func(x_dict):
    cci_score = 0
    cci_score += calc_func(x_dict, list_1, 1)
    cci_score += calc_func(x_dict, list_2, 2)
    cci_score += calc_func(x_dict, list_3, 3)
    cci_score += calc_func(x_dict, list_4, 4)
    cci_score += x_dict['age_score']
    return cci_score

# reference table 가져오기 icd9 to CCI - 나중에 reference 찾아야함
cci_ref = pd.read_csv('/shl_drive/shl_home/juhha/projects/Delirium/code_paper/data/comorbidity_code.csv')

cci_diag_conversion = {
    'Myocardial Infarction': 'MI',
    'Congestive Heart Failure': 'CHF',
    'Periphral Vascular Disease': 'PVD',
    'Cerebrovascular Disease': 'Stroke',
    'Dementia': 'Dementia',
    'Chronic Pulmonary Disease': 'Pulmonary',
    'Connective Tissue Disease-Rheumatic Disease': 'Rheumatic',
    'Peptic Ulcer Disease': 'PUD',
    'Mild Liver Disease': 'LiverMild',
    'Diabetes without complications': 'DM',
    'Diabetes with complications': 'DMcx',
    'Paraplegia and Hemiplegia': 'Paralysis',
    'Renal Disease': 'Renal',
    'Cancer': 'Cancer',
    'Moderate or Severe Liver Disease': 'LiverSevere',
    'Metastatic Carcinoma': 'Mets',
    'AIDS/HIV': 'HIV'
}
cci_ref['Category'] = cci_ref['Category'].apply(lambda x: cci_diag_conversion[x])

valid_codes = [i.strip() for i in cci_ref['Code'].tolist()]
valid_codes_str = ','.join(valid_codes)
dict_code_to_cat = {i.strip().strip('\''):j for i,j in cci_ref[['Code', 'Category']].values}

# bring diag table by subject_id & icd9_code
query = query_schema + """
    SELECT *
    FROM diagnoses_icd
    WHERE subject_id IN ({}) AND icd9_code IN ({})
""".format(subject_id_all_str, valid_codes_str)
diag = pd.read_sql_query(query, con)

diag['category'] = diag['icd9_code'].apply(lambda x: dict_code_to_cat[x])
# bring valid hadm_id for diag
hadm_id_diag = ','.join([str(i) for i in diag['hadm_id'].unique().tolist()])
# admission table과 merge 해서 진단 받은 날짜 가져오기
# *** admittime이 진단 받은 날짜보다 이전이라 생각하고 가져옴
query = query_schema + """
    SELECT *
    FROM admissions
    WHERE hadm_id IN ({})
""".format(hadm_id_diag)
ref_adm = pd.read_sql_query(query, con)
# make diag table by merging to admission table
diag = diag.merge(ref_adm[['hadm_id', 'admittime']], on = ['hadm_id'], how = 'left')

#####
# Vent
query = query_schema + """
    SELECT *
    FROM chartevents
    WHERE hadm_id IN ({}) AND itemid IN (
        639, 654, 681, 682, 683, 684,224685,224684,224686
        , 445, 448, 449, 450, 1340, 1486, 1600, 224687
        , 218,436,535,444,459,224697,224695,224696,224746,224747
        , 221,1,1211,1655,2000,226873,224738,224419,224750,227187
        , 543
        , 5865,5866,224707,224709,224705,224706
        , 60,437,505,506,686,220339,224700
        , 3459
        , 501,502,503,224702
        , 223,667,668,669,670,671,672
        , 224701
        , 467, 720, 722, 223848, 223849)
""".format(hadm_id_all_raw_str)
vent = pd.read_sql_query(query, con)
vent['value'] = 1
vent = vent.rename(columns = {'value': 'vent_status'})

#####
# SOFA
"""
1. import ICU stay table
2. import sofa table
3. merge (ICU stay) and (sofa)
"""
# Icu stay
query = query_schema + """
    SELECT *
    FROM icustays
    WHERE hadm_id IN ({})
""".format(hadm_id_all_raw_str)
icu_stay = pd.read_sql_query(query, con)
# sofa
query = query_schema + """
    SELECT *
    FROM SOFA
    WHERE hadm_id IN ({})
""".format(hadm_id_all_raw_str)
sofa = pd.read_sql_query(query, con)

# merge
sofa = sofa.merge(icu_stay[['intime', 'icustay_id']], on = ['icustay_id'], how = 'left')
sofa['new_hadm_id'] = sofa['hadm_id'].astype(str) + '@' + sofa['intime'].astype(str)


#############################################################################
##### 5/7 이후
# Reason for ICU - 예전에는 services table에서 가져왔는데 지금은 caption된거를 가져옴
ref_df = pd.read_excel('/shl_drive/shl_home/juhha/projects/Delirium/code_paper/data/MIMIC_ICU_reason_KRE.xlsx')
new_dict = {
    'Post_CPR': 'Cardiovascular',
    'Post_Intervention': 'Peri_operation',
    'Post_OP': 'Peri_operation',
    'Pre_OP': 'Peri_operation'
}
ref_df['Category'] = ref_df['Category'].apply(lambda x: new_dict[x] if x in new_dict.keys() else x)

reason_dict = {i:j for i,j in ref_df[['Reason', 'Category']].values}
reason_dict_by_id = {i:j for i,j in ref_df[['new_hadm_id', 'Category']].values}

#############################################################################
# 3. Pre-processing
#    a. get scaler created from shl
numeric_scale_dict = pickle.load(open(scaler_fileloc, 'rb'))
#    b. scaling - lab, vital
######
# Vital
vital_dict = {
    'SBP': vital[vital['vital_type'] == 'SBP'],
    'DBP': vital[vital['vital_type'] == 'DBP'],
    'HR': vital[vital['vital_type'] == 'HR'],
    'SpO2': vital[vital['vital_type'] == 'SpO2']
}
for key in vital_dict.keys():
    # drop outlier
    outlier_min = outlier_dict_vital[key]['min']
    outlier_max = outlier_dict_vital[key]['max']
    vital_dict[key]['valuenum'] = vital_dict[key]['valuenum'].apply(lambda x: x if outlier_min < x < outlier_max else np.nan)
    # normalize
    normalize_mean = numeric_scale_dict['vital']['norm'][key]['mean']
    normalize_std = numeric_scale_dict['vital']['norm'][key]['std']
    vital_dict[key]['valuenum'] = vital_dict[key]['valuenum'].apply(lambda x: (x - normalize_mean) / normalize_std if x else x)
    # min_max
    scaler_min = numeric_scale_dict['vital']['min_max_scaler'][key]['min']
    scaler_max = numeric_scale_dict['vital']['min_max_scaler'][key]['max']
    vital_dict[key]['valuenum'] = vital_dict[key]['valuenum'].apply(lambda x: (x - scaler_min) / (scaler_max - scaler_min) if x else x)
    vital_dict[key] = vital_dict[key][vital_dict[key]['valuenum'].isnull() == False]
temp_cols = list(vital_dict.keys())
vital = vital_dict[temp_cols[0]]
for col in temp_cols[1:]:
    vital = pd.concat((vital, vital_dict[col]), axis = 0, sort = False)
    
######
# Lab
lab = lab[['hadm_id', 'itemid', 'charttime', 'valuenum', 'valueuom', 'lab_name', 'lab_code']]
lab_dict = {}
for lab_code in lab['lab_code'].unique().tolist():
    lab_dict[lab_code] = lab[lab['lab_code'] == lab_code]
    # normalize
    mean_ = numeric_scale_dict['lab']['norm'][lab_code]['mean']
    std_ = numeric_scale_dict['lab']['norm'][lab_code]['std']
    lab_dict[lab_code]['valuenum'] = (lab_dict[lab_code]['valuenum'] - mean_) / std_
    # min max scaler
    min_ = numeric_scale_dict['lab']['min_max_scaler'][lab_code]['min']
    max_ = numeric_scale_dict['lab']['min_max_scaler'][lab_code]['max']
    lab_dict[lab_code]['valuenum'] = lab_dict[lab_code]['valuenum'].apply(lambda x: (x - min_) / (max_ - min_))
    lab_dict[lab_code] = lab_dict[lab_code][lab_dict[lab_code]['valuenum'].isnull() == False]
    
temp_cols = list(lab_dict.keys())
lab = lab_dict[temp_cols[0]]
for col in temp_cols[1:]:
    lab = pd.concat((lab, lab_dict[col]), axis = 0, sort = False)
    
#############################################################################
# 4. Merge
print('Start merging...')
##########################################
# transfer
"""
Transfer
    ref_cols: [subject_id, hadm_id, intime]
    key_col: [new_hadm_id*]
    cat_col: [ward_CD*]
    * new_hadm_ID: (hadm_ID + intime)
    ** ward_CD: "transfers" table에서 "curr_careunit"컬럼명. "ward_CD랑" mapped
"""
# bring ward_CD
transfer_cols = ['subject_id', 'hadm_id', 'new_hadm_id', 'intime', 'case_control', 'ward_CD']
df = transfers[transfer_cols]
# cat_cols 더미화
df = pd.get_dummies(df, columns = ['ward_CD'])
print('transfer: {}'.format(df.shape))
##########################################
# patients
"""
Patients
    Table: Patients
    ref_cols: [subject_id]
    cat_cols: [gender]
    num_cols: [age*]
    * age: dob와 intime 가지고 계산
"""
# bring gender and age
patients_cols = ['subject_id', 'gender', 'dob']
patients = patients[patients_cols]
# subject_id 가지고 merge
temp = df.merge(patients, on = ['subject_id'])
# age 가져오기
temp['age'] = temp[['intime', 'dob']].apply(lambda x: (x[0].year - x[1].year) - ((x[0].month,x[1].day) < (x[0].month, x[1].day)), axis = 1)
select_cols = ['subject_id', 'hadm_id', 'new_hadm_id', 'intime', 'case_control', 'ward_CD', 'gender', 'age']
# cat_col categorize
temp = pd.get_dummies(temp, columns = ['gender'])
df = temp.drop(columns = ['dob'])
print('patients: {}'.format(df.shape))
##########################################
# sleep
"""
sleep
    Table: chartevents
    ref_cols: [hadm_id, charttime]
    cat_cols: [sleep_disorder*]
    * sleep_disorder: "chartevents" table에서 "value" 칼럼. 0 or 1
"""
# merge
temp = df[['hadm_id', 'new_hadm_id', 'intime']].merge(sleep[['hadm_id', 'charttime', 'value']])
temp['time_took'] = (temp['charttime'] - temp['intime']).astype(int).apply(lambda x: x / hour)
# bring only valid by time
temp['valid'] = temp['time_took'].apply(lambda x: -cutoff_hour <= x <= diff_hour)
temp['value'] = temp['value'].apply(lambda x: x[0])
temp = temp[temp['valid']]
temp = temp.sort_values(['charttime'])
temp = temp.drop_duplicates(['new_hadm_id'], keep = 'last')
# cat_col categorize
temp = pd.get_dummies(temp[['new_hadm_id', 'value']].rename(columns = {'value': 'sleep_disorder'}), columns = ['sleep_disorder'])
df = df.merge(temp, on = ['new_hadm_id'], how = 'left')
print('sleep: {}'.format(df.shape))


# ##########################################
# # 5/7 이전에는 ICU_in_reason_NM을 services에서 가져왔는데 5/7이후에는 아님
# # services
# """
# services
#     Table: services, transfers
#     ref_col: [new_hadm_id**]
#     cat_cols: [ICU_in_reason_NM*]
#     * ICU_in_reason_NM: "services" table 에서 "curr_service" 칼럼. "ICU_in_reason_NM"과 mapped
#     ** 1차 pre-processing 할때 애초에 merge해서 "new_hadm_id"를 부여
# """
# services_cols = ['new_hadm_id', 'ICU_in_reason_NM']
# services = services[services_cols]
# temp = df.merge(services, on = ['new_hadm_id'], how = 'left')
# df = temp
# df = pd.get_dummies(df, columns = ['ICU_in_reason_NM'])
# print('services: {}'.format(df.shape))

##########################################
# ICU_in_reason_NM
temp = df['new_hadm_id'].apply(lambda x: reason_dict_by_id[x])
df['ICU_in_reason_NM'] = df['new_hadm_id'].apply(lambda x: reason_dict_by_id[x])
df = pd.get_dummies(df, columns = ['ICU_in_reason_NM'])
print('ICU_in_reason_NM: {}'.format(df.shape))

##########################################
# vital
"""
vital
    Table: chartevents
    ref_col: [hadm_id, charttime]
    stat_cols*: ['SBP', 'DBP', 'SpO2', 'HR']
    * stat_cols: 하나의 prime-key에 여러가지 numeric가 부여될수 있을 경우 통계적 정보를 가져옴 
"""
# bring valid data
# valid by hadm_id
temp = df[['hadm_id', 'new_hadm_id', 'intime']].merge(vital, on = ['hadm_id'], how = 'inner')
# valid by (-cutoff_hour ~ record_DT ~ t1)
temp['time_took'] = (temp['charttime'] - temp['intime']).astype(int).apply(lambda x: x / hour)
# bring only -48 ~ charttime ~ t1
temp['valid'] = temp['time_took'].apply(lambda x: -cutoff_hour <= x <= diff_hour)
temp = temp[temp['valid']]
# bring stat values
temp = temp.sort_values(['charttime'])
temp = temp[['new_hadm_id', 'vital_type', 'valuenum']]
vital_dict = {
    'SBP': temp[temp['vital_type'] == 'SBP'].groupby(['new_hadm_id']).agg(['mean','max','min','std', 'last']).reset_index(),
    'DBP': temp[temp['vital_type'] == 'DBP'].groupby(['new_hadm_id']).agg(['mean','max','min','std', 'last']).reset_index(),
    'SpO2': temp[temp['vital_type'] == 'SpO2'].groupby(['new_hadm_id']).agg(['mean','max','min','std', 'last']).reset_index(),
    'HR': temp[temp['vital_type'] == 'HR'].groupby(['new_hadm_id']).agg(['mean','max','min','std', 'last']).reset_index()
}
for key in vital_dict.keys():
    vital_dict[key].columns = [i if j == '' else key + '_' + j for i,j in vital_dict[key].columns]
temp_cols = list(vital_dict.keys())
temp = vital_dict[temp_cols[0]]
for col in temp_cols[1:]:
    temp = temp.merge(vital_dict[col], on = ['new_hadm_id'], how = 'outer')
# merge
df = df.merge(temp, on = ['new_hadm_id'], how = 'left')
print('vital: {}'.format(df.shape))
##########################################
# gcs
"""
gcs
    Table: chartevents
    ref_col: [hadm_id, charttime]
    stat_cols: ['E','V','M']
"""
# bring gcs
gcs_cols = ['hadm_id', 'itemid', 'charttime', 'valuenum']
gcs = gcs[gcs_cols]
# valid hadm_id
temp = df[['hadm_id', 'new_hadm_id', 'intime']].merge(gcs, on = ['hadm_id'], how = 'inner')
# valid by (-cutoff_hour ~ record_DT ~ t1)
temp['time_took'] = (temp['charttime'] - temp['intime']).astype(int).apply(lambda x: x / hour)
# bring only -24 ~ charttime ~ t1
temp['valid'] = temp['time_took'].apply(lambda x: -gcs_cutoff_hour <= x <= diff_hour)
temp = temp[temp['valid']]
# stat values
temp = temp.sort_values(['charttime'])
temp = temp[['new_hadm_id', 'itemid', 'valuenum']]
gcs_dict = {
    'E': temp[temp['itemid'] == 'E'].groupby(['new_hadm_id']).agg(['mean','max','min','std','last']).reset_index(),
    'V': temp[temp['itemid'] == 'V'].groupby(['new_hadm_id']).agg(['mean','max','min','std','last']).reset_index(),
    'M': temp[temp['itemid'] == 'M'].groupby(['new_hadm_id']).agg(['mean','max','min','std','last']).reset_index()
}
for key in gcs_dict.keys():
    gcs_dict[key].columns = [i if j == '' else key + '@' + j for i,j in gcs_dict[key].columns]
temp_cols = list(gcs_dict.keys())
temp = gcs_dict[temp_cols[0]]
for col in temp_cols[1:]:
    temp = temp.merge(gcs_dict[col], on = ['new_hadm_id'], how = 'outer')
# merge
df = df.merge(temp, on = ['new_hadm_id'], how = 'left')
print('gcs: {}'.format(df.shape))
#############################################
"""
DRUG
    Table: prescriptions
    ref_cols: [hadm_id, startdate]
    binary_cols*: [MED1, MED2, ..., MED_m]
    * binary_cols: 여기서 binary_cols는 사실 medication_type의 dummy 변수들임. medication_type은 pre-processing 할때 미리 대분류를 시켜놓음 (항생제, ...)
    ** (-cutoff_hour ~ t1) 사이 처방 받은 적이 있다/없다
    *** note: 날짜로만 나와 있어서 이틀 전 ~ 해당일-1 이렇게 가져옴
"""
# bring drug
# valid hadm_id
temp = df[['hadm_id', 'new_hadm_id', 'intime']].merge(drug, on = ['hadm_id'], how = 'inner')
# valid by (-cutoff_hour ~ record_DT ~ t1)
temp['intime'] = pd.to_datetime(temp['intime'].apply(lambda x: x + timedelta(hours = diff_hour)).astype(str).apply(lambda x: x[:10]))
temp['valid'] = (temp['intime'] - temp['startdate']).apply(lambda x: -2 <= x.days < 0)
temp = temp[temp['valid']]
# yes/no - duplicate 필요 없음
temp = temp.drop_duplicates(['new_hadm_id', 'drug'])
temp['drug'] = temp['drug'].astype('category')
temp = temp[['new_hadm_id', 'drug']]
# dummy화 - binary_cols
temp = pd.get_dummies(temp, columns = ['drug'], prefix = "", prefix_sep = '')
temp = temp.groupby(['new_hadm_id']).agg('max').reset_index()
# merge
df = df.merge(temp, on = ['new_hadm_id'], how = 'left')
print('drug: {}'.format(df.shape))
#############################################
# diag (CCI)
"""
diag
    Table: diagnoses_icd
    ref_cols: [hadm_id, admittime*]
    binary_cols**: [diag1, diag2, ..., diag_m]
    numeric_cols: [age_score, CCI]
    * "diagnoses_icd" table에서 "admittime" 변수와 "intime"을 ref해서 해당 병명이 있다/ 없다 가져옴
    ** binary_cols: 여기서 binary_cols는 사실 diag_type의 dummy 변수들임. diag_type변수는 pre-processing 할때 미리 대분류를 시켜놓음
    *** CCI와 대분류는 인터넷상에 reference할 테이블 찾아서 그 기준으로 함
"""
# bring valid lab data
# valid hamd_id
temp = df.merge(diag, on = ['hadm_id'], how = 'inner')
temp = df[['hadm_id', 'new_hadm_id', 'intime']].merge(diag[['hadm_id', 'category', 'admittime']], on = ['hadm_id'], how = 'inner')
# valid by (admittime ~ intime)
temp['valid'] = temp['intime'] >= temp['admittime']
temp = temp[temp['valid']]
# dummy화
temp = temp[['new_hadm_id', 'category']]
temp = pd.get_dummies(temp, columns = ['category'], prefix = '', prefix_sep = '').groupby(['new_hadm_id']).agg('max').reset_index()
# CCI & age_score 구하기
df = df.merge(temp, on = ['new_hadm_id'], how = 'left')
df['age_score'] = df['age'].apply(lambda x: get_age_score(x))
df['CCI'] = df.apply(lambda x: cci_calculation_func(x), axis = 1)
print('diag: {}'.format(df.shape))
#############################################
# lab
"""
lab
    Table: labevents
    ref_cols: [hadm_id, charttime]
    stat_cols: [lab1_valuenum, lab2_valuenum, ...]
    * lab_type은 pre-processing할때 미리 code로 mapping시켜놓음
    ** scaling: normalization, min_max by 삼성서울병원 scaler
"""
# bring valid lab data
# valid hadm_id
temp = df[['hadm_id', 'new_hadm_id', 'intime']].merge(lab[['hadm_id', 'charttime', 'lab_name', 'valueuom', 'valuenum', 'lab_code']], how = 'inner')
# valid by (-24*7 ~ charttime ~ t1)
temp['time_took'] = (temp['charttime'] - temp['intime']).astype(int).apply(lambda x: x / hour)
temp['valid'] = temp['time_took'].apply(lambda x: -lab_cutoff_hour <= x <= diff_hour)
# print(temp.shape)
temp = temp[temp['valid']]
# print(temp.shape)
# groupby할때 latest lab value 가져오기 위해서 chattime기준으로 sort함
temp = temp.sort_values(['charttime'])
temp = temp[['new_hadm_id', 'lab_code', 'valuenum']]
# make stat_cols for each lab_code
lab_dict = {}
for lab_code in temp['lab_code'].unique():
    lab_dict[lab_code] = temp[temp['lab_code'] == lab_code].groupby(['new_hadm_id']).agg(['mean','max','min','std','last']).reset_index()
for key in lab_dict.keys():
    lab_dict[key].columns = [i if j == '' else key + '_' + j for i,j in lab_dict[key].columns]
temp_cols = list(lab_dict.keys())
temp = lab_dict[temp_cols[0]]
for col in temp_cols[1:]:
    temp = temp.merge(lab_dict[col], on = ['new_hadm_id'], how = 'outer')
# merge
df = df.merge(temp, on = ['new_hadm_id'], how = 'left')
print('lab: {}'.format(df.shape))

#############################################
# 5/7 이후
# vent
temp = df[['hadm_id', 'new_hadm_id', 'intime']].merge(vent[['hadm_id', 'charttime']])
temp['time_took'] = (temp['charttime'] - temp['intime']).astype(int).apply(lambda x: x / hour)
temp['valid'] = temp['time_took'].apply(lambda x: -4 <= x <= 0)
temp['vent_status'] = 1
temp = temp[temp['valid']]
temp = temp.sort_values(['charttime'])
temp = temp.drop_duplicates(['new_hadm_id'], keep = 'last')
# temp = pd.get_dummies(temp[['new_hadm_id', 'vent_status']], columns = ['vent_status'])
temp = temp[['new_hadm_id', 'vent_status']]
df = df.merge(temp, on = ['new_hadm_id'], how = 'left')
print('vent: {}'.format(df.shape))


print('Done merging...')

df.to_csv(save_fileloc, index = False)
print('Saved mimic input at {}'.format(save_fileloc))