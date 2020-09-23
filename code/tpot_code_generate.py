import sys, os
import pandas as pd

from tpot import TPOTClassifier

# Configuration
base_dir = os.path.join('/'.join(os.path.abspath(__file__).split('/')[:-1]), '../dataset')
tpot_export_dir = os.path.join(base_dir, 'result', 'tpot_code_result')

# import data
df = pd.read_csv(os.path.join(base_dir, 'input_data', 'input_smc.csv'))
train_df = df[(df['ward_in_DT'].apply(lambda x: x[:4] != '2019')) & (df['age'] >= 18)]
test_df = df[(df['ward_in_DT'].apply(lambda x: x[:4] == '2019')) & (df['age'] >= 18)]
list_features = ['ward_CD_Internal Medicine Group', 'ward_CD_Surgical Medicine Group', 'vent_status', 'V@min', 'steroids', 'SpO2_mean', 'SBP_std', 'RR_min', 'Parasympathomimetic (Cholinergic) agents', 'Opiate Antagonists', 'Opiate Agonists', 'Miscellaneous Antidepressants', 'miscellaneous_antidepressants_anxiolytics_sedatives_hypnotics', 'M@min', 'Inotropics', 'ICU_in_reason_NM_Respiratory', 'ICU_in_reason_NM_Peri_operation', 'ICU_in_reason_NM_Cardiovascular', 'ICU_in_reason_NM_Nephrology', 'ICU_in_reason_NM_Neurology', 'ICU_in_reason_NM_Severe_Trauma', 'ICU_in_reason_NM_Metabolic_Endocrine', 'ICU_in_reason_NM_Hematology', 'ICU_in_reason_NM_Gastro_Intestinal', 'HR_last', 'gender_F', 'gender_M', 'E@min', 'DBP_last', 'CCI', 'BL319301_last', 'BL318512_max', 'BL318507_mean', 'BL318504_mean', 'BL318503_last', 'BL318502_max', 'BL318501_last', 'BL3157_last', 'BL3140_last', 'BL3138_max', 'BL3137_last', 'BL3132_mean', 'BL3131_last', 'BL3123_min', 'BL3120_last', 'BL3119_max', 'BL3118_mean', 'BL3116_min', 'BL3115_mean', 'BL3114_last', 'BL3112_mean', 'BL3111_max', 'BL2112_last', 'BL211103_mean', 'BL2021_mean', 'BL2016_last', 'BL2014_max', 'BL2013_max', 'BL2011_min', 'benzodiazepines', 'anticholinergic_antipsychotics', 'Antibiotics', 'age']
trX = train_df[list_features].values
trY = train_df['case_control'].apply(lambda x: 1 if x == 'case' else 0).values.reshape(-1,1)
teX = test_df[list_features].values
teY = test_df['case_control'].apply(lambda x: 1 if x == 'case' else 0).values.reshape(-1,1)

# tpot generator
tpot = TPOTClassifier()

print(tpot)
tpot.fit(trX, trY)
auc = tpot.score(teX, teY)
tpot.export(os.path.join(tpot_export_dir, 'tpot_result.py'))