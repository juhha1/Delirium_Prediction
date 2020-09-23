import pandas as pd
import numpy as np
import pickle, os, joblib
from sklearn.metrics import auc, roc_curve, confusion_matrix
import tensorflow as tf

base_dir = os.path.join('/'.join(os.path.abspath(__file__).split('/')[:-1]), '../dataset')
ml_dir = os.path.join(base_dir, 'ml_model')
data_dir = os.path.join(base_dir, 'input_data')
save_dir = os.path.join(base_dir, 'result')

df = pd.read_csv(os.path.join(data_dir, 'input_mimic.csv'))

list_features = ['ward_CD_Internal Medicine Group', 'ward_CD_Surgical Medicine Group', 'vent_status', 'V@min', 'steroids', 'SpO2_mean', 'SBP_std', 'RR_min', 'Parasympathomimetic (Cholinergic) agents', 'Opiate Antagonists', 'Opiate Agonists', 'Miscellaneous Antidepressants', 'miscellaneous_antidepressants_anxiolytics_sedatives_hypnotics', 'M@min', 'Inotropics', 'ICU_in_reason_NM_Respiratory', 'ICU_in_reason_NM_Peri_operation', 'ICU_in_reason_NM_Cardiovascular', 'ICU_in_reason_NM_Nephrology', 'ICU_in_reason_NM_Neurology', 'ICU_in_reason_NM_Severe_Trauma', 'ICU_in_reason_NM_Metabolic_Endocrine', 'ICU_in_reason_NM_Hematology', 'ICU_in_reason_NM_Gastro_Intestinal', 'HR_last', 'gender_F', 'gender_M', 'E@min', 'DBP_last', 'CCI', 'BL319301_last', 'BL318512_max', 'BL318507_mean', 'BL318504_mean', 'BL318503_last', 'BL318502_max', 'BL318501_last', 'BL3157_last', 'BL3140_last', 'BL3138_max', 'BL3137_last', 'BL3132_mean', 'BL3131_last', 'BL3123_min', 'BL3120_last', 'BL3119_max', 'BL3118_mean', 'BL3116_min', 'BL3115_mean', 'BL3114_last', 'BL3112_mean', 'BL3111_max', 'BL2112_last', 'BL211103_mean', 'BL2021_mean', 'BL2016_last', 'BL2014_max', 'BL2013_max', 'BL2011_min', 'benzodiazepines', 'anticholinergic_antipsychotics', 'Antibiotics', 'age']

# define bins of thresholds to analyze
thres = np.arange(0,1,0.01)

# external validation
list_clf = ['xgb.joblib', 'lr.joblib', 'nn.h5', 'rf.joblib']
list_auc, list_tn, list_fp, list_fn, list_tp = [], [], [], [], []
list_ml_name = []
list_thres = []

def make_input_mimic(df_mimic, list_cols):
    output_x = []
    for col in list_cols:
        if col in df_mimic.columns:
            output_x.append(df_mimic[col].values.tolist())
        else:
            output_x.append([0] * df_mimic.shape[0])
    return np.array(output_x).T

for i, clf in enumerate(list_clf):
    ml_name = clf
    valid_list_features = [col for col in list_features if col in df.columns]
    X = make_input_mimic(df.fillna(0), list_features)
    Y = df['case_control'].apply(lambda x: 1 if x == 'case' else 0).values
    if clf != 'nn.h5':
        print(os.path.join(ml_dir, ml_name))
        clf = joblib.load(os.path.join(ml_dir, ml_name))
        probas = clf.predict_proba(X)[:,1]
    else:
        clf = tf.keras.models.load_model(os.path.join(ml_dir, ml_name))
        probas = clf.predict(X)
    fpr, tpr, _ = roc_curve(y_true = Y, y_score = probas)
    auc_score = auc(fpr, tpr)
    for t in thres:
        list_thres.append(t)
        pred = (probas >= t) * 1
        tn, fp, fn, tp = confusion_matrix(y_true = Y, y_pred = pred).ravel()
        list_auc.append(auc_score); list_tn.append(tn); list_fp.append(fp); list_fn.append(fn); list_tp.append(tp)
        list_ml_name.append(ml_name)

output_df = pd.DataFrame({
    'ml_name': list_ml_name,
    'thres': list_thres,
    'auc': list_auc,
    'tp': list_tp,
    'fn': list_fn,
    'tn': list_tn,
    'fp': list_fp
})

output_df['sensitivity'] = output_df['tp'] / (output_df['tp'] + output_df['fn'])
output_df['specificity'] = output_df['tn'] / (output_df['tn'] + output_df['fp'])
output_df['ppv (precision)'] = output_df['tp'] / (output_df['tp'] + output_df['fp'])
output_df['npv'] = output_df['tn'] / (output_df['tn'] + output_df['fn'])
output_df['f1'] = 2 * (output_df['ppv (precision)'] * output_df['sensitivity']) / (output_df['ppv (precision)'] + output_df['sensitivity'])
output_df.to_excel(os.path.join(save_dir, 'mimic_external_validation.xlsx'))