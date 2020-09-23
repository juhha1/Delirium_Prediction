import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from imblearn.under_sampling import RandomUnderSampler
import pickle, joblib, datetime, os
import tensorflow as tf

#############################################################################
# Configuration
base_dir = os.path.join('/'.join(os.path.abspath(__file__).split('/')[:-1]), '../dataset')

config = {
    "input_fileloc": os.path.join(base_dir, 'input_data', 'input_smc.csv'),
    "save_fileloc": os.path.join(base_dir, 'ml_model'),
    "list_features": ['ward_CD_Internal Medicine Group', 'ward_CD_Surgical Medicine Group', 'vent_status', 'V@min', 'steroids', 'SpO2_mean', 'SBP_std', 'RR_min', 'Parasympathomimetic (Cholinergic) agents', 'Opiate Antagonists', 'Opiate Agonists', 'Miscellaneous Antidepressants', 'miscellaneous_antidepressants_anxiolytics_sedatives_hypnotics', 'M@min', 'Inotropics', 'ICU_in_reason_NM_Respiratory', 'ICU_in_reason_NM_Peri_operation', 'ICU_in_reason_NM_Cardiovascular', 'ICU_in_reason_NM_Nephrology', 'ICU_in_reason_NM_Neurology', 'ICU_in_reason_NM_Severe_Trauma', 'ICU_in_reason_NM_Metabolic_Endocrine', 'ICU_in_reason_NM_Hematology', 'ICU_in_reason_NM_Gastro_Intestinal', 'HR_last', 'gender_F', 'gender_M', 'E@min', 'DBP_last', 'CCI', 'BL319301_last', 'BL318512_max', 'BL318507_mean', 'BL318504_mean', 'BL318503_last', 'BL318502_max', 'BL318501_last', 'BL3157_last', 'BL3140_last', 'BL3138_max', 'BL3137_last', 'BL3132_mean', 'BL3131_last', 'BL3123_min', 'BL3120_last', 'BL3119_max', 'BL3118_mean', 'BL3116_min', 'BL3115_mean', 'BL3114_last', 'BL3112_mean', 'BL3111_max', 'BL2112_last', 'BL211103_mean', 'BL2021_mean', 'BL2016_last', 'BL2014_max', 'BL2013_max', 'BL2011_min', 'benzodiazepines', 'anticholinergic_antipsychotics', 'Antibiotics', 'age'],

    "ml_models":["rf", "xgb", "lr", "nn"],
    "lr":{
        "n_jobs": -1,
        "max_iter": 1000,
        "random_state": 1
    },
    "rf":
    {
        "n_estimators":1000,
        "n_jobs":-1,
        "random_state": 67
    },
    "xgb":
    {
        "learning_rate":0.1,
        "max_depth":6,
        "min_child_weight":2,
        "n_estimators":100,
        "subsample":0.7500000000000001,
        "n_jobs":-1,
        "random_state": 38
    },
    "nn":
    {
        "hidden_units":[512,256,128],
        "output_activation":"sigmoid",
        "loss":"binary_crossentropy",
        "optimizer":"sgd",
        "epoch":50
    },
    "v_type_list": ["after_5_7"],
    
    "train_all": 1,
    "sampling": 0,
    "sampling_random_state": 0,
    "split_by_year": "by_year",
    "label_type": "case_control",

    "selective_features": 0,
    "try_only_selective": 0,
    "list_selective_features": {}
}
#############################################################################
# Functions for generating model
def Model_RandomForest(config):
    return RandomForestClassifier(**config)

def Model_LogisticRegression(config):
    return LogisticRegression(**config)

def Model_XGB(config):
    return XGBClassifier(**config)

def Model_NN(config):
    hidden_units = config.pop('hidden_units') if 'hidden_unit' in config else [512,256,128,64]
    output_activation = config.pop('output_activation') if 'output_activation' in config else 'sigmoid'
    loss = config['loss'] if 'loss' in config else 'binary_crossentropy'
    optimizer = config['optimizer'] if 'optimizer' in config else 'adam'
    input_dim = config['input_dim']
    output_dim = config['output_dim']
    
    Inputs = tf.keras.layers.Input(shape = (input_dim))
    model = tf.keras.layers.Dense(hidden_units[0], activation = 'relu')(Inputs)
    for h_unit in hidden_units[1:]:
        model = tf.keras.layers.Dense(h_unit, activation= 'relu')(model)
    model = tf.keras.layers.Dense(output_dim, activation = output_activation)(model)
    model = tf.keras.Model(Inputs, model)
    model.compile(loss = loss, optimizer = optimizer)
    return model

def import_input(config):
    data_path = config.pop('data_path')
    read_dict = {'pkl':pd.read_pickle, 'csv': pd.read_csv, 'xlsx': pd.read_excel}
    data_type = data_path.split('.')[-1]
    df = read_dict[data_type](data_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns = ['Unnamed: 0'])
    df['ward_in_DT'] = pd.to_datetime(df['ward_in_DT'])
    df.fillna(0, inplace = True)
    return df

def train_test_df(df, config):
    split = config['split']
    label_type = config['label_type']
    
    if 0 not in df[label_type].unique():
        df[label_type] = df[label_type].apply(lambda x: 1 if x.lower() == 'case' else 0)
    
    if split == 'by_year':
        train_df = df[df['ward_in_DT'].apply((lambda x: x.year < 2019)) & (df['age'] >= 18)]
        test_df = df[df['ward_in_DT'].apply((lambda x: x.year == 2019)) & (df['age'] >= 18)]
    else:
        train_df, test_df = train_test_split(df, test_size = 0.2)
    return train_df, test_df
    
def make_input(train_df, test_df, l_features, config, label_type):
    trX = train_df[l_features].astype(float).values
    teX = test_df[l_features].astype(float).values
    trY = train_df[label_type].astype(float).values
    teY = test_df[label_type].astype(float).values
    return (trX, trY), (teX, teY)    

def train(config, trX, trY):
    model_function_dict = {'lr': Model_LogisticRegression,
                      'rf': Model_RandomForest,
                      'xgb':Model_XGB,
                      'nn':Model_NN}
    list_models = config['ml_models']
    clfs_dict = {model: None for model in list_models}
    for model in list_models:
        model_config = config[model] if model in config else {}
        if model == 'nn':
            model_config['input_dim'] = trX.shape[1]
            model_config['output_dim'] = 1
        clf = model_function_dict[model](model_config)
        if model == 'nn':
            epoch = model_config['epoch'] if 'epoch' in model_config else 10
            clf.fit(trX, trY, epochs = epoch, verbose = 0)
        else:
            clf.fit(trX, trY)
        clfs_dict[model] = clf
    return clfs_dict

#############################################################################
# Make models
df = pd.read_csv(config['input_fileloc'])
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns = ['Unnamed: 0'])
df['ward_in_DT'] = pd.to_datetime(df['ward_in_DT'])
l_features = config['list_features']

if config['train_all']:
    X = df[l_features].fillna(0).values
    Y = df[config['label_type']].apply(lambda x: 1 if x == 'case' else 0).values
else:
    train_df = df[df['ward_in_DT'].apply(lambda x: x.year < 2019)]
    X = train_df[l_features].fillna(0).values
    Y = train_df[config['label_type']].apply(lambda x: 1 if x == 'case' else 0).values
if config['sampling']:
    rus = RandomUnderSampler(random_state = config['sampling_random_state'])
    X,Y = rus.fit_resample(X,Y)
clfs = train(config,X,Y)
for ml in config['ml_models']:
    clf_dict = {
        'list_features': l_features,
        'model': clfs[ml]
    }
    today = str(datetime.datetime.now())[5:10].replace('-','_')
    if ml == 'nn':
#         save_filename = '{}-{}-train_all_{}-v_type_{}-sampling_{}-seed_{}.h5'.format(today, ml, config['train_all'], 'after_5_7', config['sampling'], 'none')
        save_filename = '{}.h5'.format(ml)
        clfs[ml].save(os.path.join(config['save_fileloc'], save_filename))
    else:
#         save_filename = '{}-{}-train_all_{}-v_type_{}-sampling_{}-seed_{}.pkl'.format(today, ml, config['train_all'], 'after_5_7', config['sampling'], clfs[ml].random_state)
        save_filename = '{}.joblib'.format(ml)
        joblib.dump(clfs[ml], os.path.join(config['save_fileloc'], save_filename))
    print('Saved {}'.format(os.path.join(config['save_fileloc'], save_filename)))
    print('-------------------------')