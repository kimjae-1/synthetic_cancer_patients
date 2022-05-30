import numpy as np
import pandas as pd
import os
import shutil

from tqdm.auto import tqdm
import itertools
import random
import pickle

from fastNLP import DataSet
from missingprocessor import Processor

def cohort_selection(df):
    """
    (1) CANCER_PT_BSNF : BSPT_FRST_DIAG_YMD
    (2) CANCER_PT_BSNF : (BSPT_FRST_OPRT_YMD, BSPT_FRST_TRTM_STRT_YMD)
    0 <= Diff = MIN((2)) - (1) <= 3 months
    """
    selected_cols = ['PT_SBST_NO', 'BSPT_FRST_DIAG_YMD', 'BSPT_FRST_OPRT_YMD', 'BSPT_FRST_ANCN_TRTM_STRT_YMD','BSPT_FRST_RDT_STRT_YMD']
    df = df[selected_cols].copy()
    df['BSPT_FRST_MIN_YMD'] = df.loc[:, selected_cols[2:4]].min(axis=1)
    #df = df.replace(r'\N', np.nan)
    df[df.columns[1:]] = df[df.columns[1:]].apply(lambda x : pd.to_datetime(x, format='%Y%m%d'))

    df['BSPT_FRST_DIFF'] = (df['BSPT_FRST_MIN_YMD'] - df['BSPT_FRST_DIAG_YMD']).dt.days
    df = df[(df['BSPT_FRST_DIFF']>= 0) & (df['BSPT_FRST_DIFF'] <= 90)]

    pt_key_id = sorted(df['PT_SBST_NO'].unique())

    return pt_key_id


patient_basicinfo_path = os.path.join(os.getcwd(),'data','CLRC','clrc_pt_bsnf.csv')
df_pt_bsnf = pd.read_csv(patient_basicinfo_path, na_values='\\N')
pt_key_id = cohort_selection(df_pt_bsnf)

selected_cols = ['PT_SBST_NO', 'BSPT_IDGN_AGE', 'BSPT_SEX_CD', 'BSPT_FRST_DIAG_CD', 'BSPT_FRST_DIAG_YMD','BSPT_DEAD_YMD']
df_pt_bsnf = df_pt_bsnf[df_pt_bsnf['PT_SBST_NO'].isin(pt_key_id)][selected_cols]

df_pt_bsnf['BSPT_SEX_CD'] = df_pt_bsnf['BSPT_SEX_CD'].replace({'F': 0, 'M': 1})

diag_cd = sorted(df_pt_bsnf['BSPT_FRST_DIAG_CD'].unique())
diag_cd = {cd: i for i, cd in enumerate(diag_cd)}
df_pt_bsnf['BSPT_FRST_DIAG_CD'] = df_pt_bsnf['BSPT_FRST_DIAG_CD'].replace(diag_cd)

df_pt_bsnf['BSPT_DEAD'] = df_pt_bsnf['BSPT_DEAD_YMD'].notnull().astype(np.int32)

df_ex_diag1_raw_path = os.path.join(os.getcwd(),'data','CLRC','clrc_ex_diag1.csv')
df_ex_diag1_raw = pd.read_csv(df_ex_diag1_raw_path, encoding='cp949').replace(r'\N', np.nan)

df_ex_diag2_raw_path = os.path.join(os.getcwd(),'data','CLRC','clrc_ex_diag2.csv')
df_ex_diag2_raw = pd.read_csv(df_ex_diag2_raw_path, encoding='cp949').replace(r'\N', np.nan)

df_ex_diag1 = df_ex_diag1_raw[df_ex_diag1_raw['PT_SBST_NO'].isin(pt_key_id)]
df_ex_diag1 = df_ex_diag1[['PT_SBST_NO', 'CEXM_YMD', 'CEXM_NM', 'CEXM_RSLT_CONT', 'CEXM_RSLT_UNIT_CONT']]
df_ex_diag2 = df_ex_diag2_raw[df_ex_diag2_raw['PT_SBST_NO'].isin(pt_key_id)]
df_ex_diag2 = df_ex_diag2[['PT_SBST_NO', 'CEXM_YMD', 'CEXM_NM', 'CEXM_RSLT_CONT', 'CEXM_RSLT_UNIT_CONT']]
df_ex_diag = pd.concat([df_ex_diag1, df_ex_diag2], axis=0, ignore_index=True).sort_values(
    by=['PT_SBST_NO', 'CEXM_YMD', 'CEXM_NM']).reset_index(drop=True)

var_list = [
    'ALP',
    'ALT',
    'AST',
    'Albumin',
    'BUN',
    'Bilirubin, Total',
    'CA 19-9',
    'CEA',
    'CRP, Quantitative (High Sensitivity)',
    'ESR (Erythrocyte Sedimentation Rate)',
    'Protein, Total',
]

exclusion = ['Anti-HBs Antibody', 'Anti-HCV Antibody', 'Anti-HIV combo', 'HBsAg']
df_ex_diag = df_ex_diag[~df_ex_diag['CEXM_NM'].isin(exclusion)]

df_ex_diag = pd.merge(df_ex_diag, df_pt_bsnf[['PT_SBST_NO', 'BSPT_FRST_DIAG_YMD']],
                      how='left', on='PT_SBST_NO')
df_ex_diag[['CEXM_YMD', 'BSPT_FRST_DIAG_YMD']] = df_ex_diag[['CEXM_YMD', 'BSPT_FRST_DIAG_YMD']].apply(
    lambda x: pd.to_datetime(x, format='%Y%m%d'))

df_ex_diag['TIMESTAMP'] = (df_ex_diag['CEXM_YMD'] - df_ex_diag['BSPT_FRST_DIAG_YMD']).dt.days
df_ex_diag = df_ex_diag[(df_ex_diag['TIMESTAMP']/365 >= 0) & (df_ex_diag['TIMESTAMP']/365 <= 5)]
#df_pt_bsnf = df_pt_bsnf[df_pt_bsnf['PT_SBST_NO'].isin(df_ex_diag['PT_SBST_NO'].unique())]
df_ex_diag['CEXM_RSLT_CONT'] = df_ex_diag['CEXM_RSLT_CONT'].astype(np.float32)
cols_ex_diag = ['PT_SBST_NO', 'CEXM_NM', 'CEXM_RSLT_CONT', 'TIMESTAMP']
df_ex_diag = df_ex_diag[cols_ex_diag]

os.makedirs(os.path.join(os.getcwd(),'rtsgan-connect-data'), exist_ok = True)

data_collector = []
label_collector = []
#var_collector = []

for k, g in tqdm(df_ex_diag.groupby('PT_SBST_NO')):
    to_physionet_style = []
    
    g = g.pivot_table(index='TIMESTAMP', 
                      columns='CEXM_NM', 
                      values='CEXM_RSLT_CONT', 
                      aggfunc='mean').reset_index(drop=False)

    data_collector.append(g)
    
    age_sex_diag = df_pt_bsnf[df_pt_bsnf['PT_SBST_NO'] == k][['BSPT_IDGN_AGE', 'BSPT_SEX_CD', 'BSPT_FRST_DIAG_CD']].to_numpy()[0]
    label = df_pt_bsnf.loc[df_pt_bsnf['PT_SBST_NO']==k, 'BSPT_DEAD'].values.item()
    label_collector.append([age_sex_diag[0], age_sex_diag[1], age_sex_diag[2], label]) 

random.seed(42) ## seed works only in same cell
test_idx = sorted(random.sample(range(len(label_collector)), int(len(label_collector)*0.2)))

train_data = [data_collector[idx] for idx in range(0, len(label_collector)) if idx not in test_idx]
test_data = [data_collector[idx] for idx in test_idx]

sta = pd.DataFrame(label_collector, columns=['age', 'sex', 'diag_code', 'result'])
sta["seq_len"] = np.array([len(x) for x in data_collector])

train_sta = sta.iloc[~sta.index.isin(test_idx)].reset_index(drop=True)
test_sta = sta.iloc[test_idx].reset_index(drop=True)

dyn_train = pd.concat(train_data)

dyn_types = ['continuos'] * len(dyn_train.columns)
sta_types = ['int', 'binary', 'categorical','binary', 'int']

d_P = Processor(dyn_types, use_pri='TIMESTAMP')
s_P = Processor(sta_types)
d_P.fit(dyn_train)
s_P.fit(train_sta)

def build_dataset(sta, dyn, seq_len):
    d_lis=[d_P.transform(ds) for ds in dyn] #dataframe to array len=6
    d = [x[0].tolist() for x in d_lis]
    lag = [x[1].tolist() for x in d_lis]
    mask = [x[2].tolist() for x in d_lis]
    times = [x[-1].tolist() for x in d_lis] 
    priv = [x[3].tolist() for x in d_lis]
    nex = [x[4].tolist() for x in d_lis]
    
    s = s_P.transform(sta)
    label = [float(x[-2]) for x in s] #-1=seq_len, -2=result(death=1)
    
    dataset = DataSet({"seq_len": seq_len, 
                       "dyn": d, "lag":lag, "mask": mask,
                       "sta": s, "times":times, "priv":priv, "nex":nex, "label": label
                      })
    return dataset

train_set = build_dataset(train_sta, train_data, train_sta['seq_len'].tolist())
test_set = build_dataset(test_sta, test_data, test_sta['seq_len'].tolist())

finaldic = {
    "train_set": train_set,
    'raw_set': (train_sta, train_data),
    'test_set': (test_sta, test_data),
    'val_set': test_set,
    "dynamic_processor": d_P,
    "static_processor":s_P
}

with open(os.path.join(os.getcwd(),'rtsgan-connect-data/connect_clrc.pkl'), "wb") as f:
    pickle.dump(finaldic, f)

