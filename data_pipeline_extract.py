#%%
"""
전북대학교병원 데이터 반출
"""


#%%
## >>> [TMP] >>>
import sys, os
sys.path.append('/home/mincheol/git/synthetic_cancer_patients')
os.chdir('/home/mincheol/git/synthetic_cancer_patients')
## <<< [TMP] <<<


#%%
import os

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


#%%
# CLRC
filepath_data = os.path.join('/', 'home', 'mincheol', 'ext', 'hdd1', 'data', 'CONNECT_extract', 'CLRC_raw')
filepath_out = os.path.join('/', 'home', 'mincheol', 'ext', 'hdd1', 'data', 'CONNECT_extract', 'CLRC')
encoding = 'CP949'

df_key1_raw = pd.read_csv(os.path.join(filepath_data, 'clrc_trtm_casb.csv'), na_values='\\N', encoding=encoding)

condition1 = [
    'Palliative mFOLFIRI(Irinotecan/LV/5-FU) for Colorectal Cancer',
    'Avastin + FOLFIRI for colorectal ca (1st, 2nd line)',
    'Palliative 1st line Cetuximab  + FOLFIRI (biweekly)',
    'Avastin + mFOLFOX for colorectal ca (1st,2nd line)',
    'Adjuvant FOLFOX for CRC',
    'Palliative mFOLFOX(Oxaliplatin/LV/5-FU) for Colorectal Cancer',
    'Avastine + mFOLFOX for colorectal ca (2nd line)',
    'Cetuximab + FOLFOX  for Colorectal Cancer',
    'Avastin/FOLFOX for colorectal ca (1st,2nd line)',
    'FOLFIRI for colorectal ca (1st, 2nd line)',
    'Avastine + mFOLFOX for colorectal ca (1st,2nd line)',
    'Palliative FOLFIRI for Colorectal Cancer',
    'Avastin + FOLFIRI for colorectal ca',
    'Avastine + mFOLFOX for colorectal ca',
    'Palliative mFOLFOX(Oxaliplatin/LV/5-FU)/Avastin for Colorectal Cancer',
    '[FOLFIRI/Bevacizumab] vs [XELIRI/Bevacizumab] in colorectal ca (2nd line), Phase III',
    'Cetuximab + FOLFIRI in Colorectal Cancer (1st line)',
    'Avastine + FOLFIRI for colorectal ca (1st line)',
    'Avastine + FOLFIRI for colorectal ca (1st, 2nd line)',
    'Cetuximab biweekly + mFOLFOX for colorectal ca (1st,2nd line)',
    'Avastin + mFOLFOX for colorectal cancer',
    'Adjuvant FOLFOX',
    'Avastin + FOLFIRI for Colorectal Cancer',
    'Cetuximab + FOLFOX-4 for Colorectal Cancer ',
    'FOLFOX  for Colorectal Cancer',
    'Cetuximab + FOLFIRI (1st line, biweekly)',
    'FOLFOX for for Colorectal Cancer',
    'Neoadjuvant FOLFOX for CRC ',
    'Palliative FOLFOX for CRC',
    'Palliative mFOLFOX(Oxaliplatin/LV/5-FU) ★oxalitin 비급여★',
    ' FOLFIRI',
]
key_id_1 = df_key1_raw[df_key1_raw['CSTR_REGN_CD_ETC_CONT'].isin(condition1)]['PT_SBST_NO'].unique()

df_key2_raw = pd.read_csv(os.path.join(filepath_data, 'clrc_pt_bsnf.csv'), na_values='\\N', encoding=encoding)

condition2 = 18
df_key2 = df_key2_raw[df_key2_raw['PT_SBST_NO'].isin(key_id_1)]
key_id_2 = df_key2[df_key2['BSPT_IDGN_AGE'] >= condition2]['PT_SBST_NO'].unique()

for filename in sorted(os.listdir(filepath_data)):
    df_raw = pd.read_csv(os.path.join(filepath_data, filename), na_values='\\N', encoding=encoding)
    df = df_raw[df_raw['PT_SBST_NO'].isin(key_id_2)]

    print(filename, len(df_raw), len(df))
    df.to_csv(os.path.join(filepath_out, filename), na_rep='\\N', encoding=encoding, index=False)


#%%
# LUNG
filepath_data = os.path.join('/', 'home', 'mincheol', 'ext', 'hdd1', 'data', 'CONNECT_extract', 'LUNG_raw')
filepath_out = os.path.join('/', 'home', 'mincheol', 'ext', 'hdd1', 'data', 'CONNECT_extract', 'LUNG')
encoding = 'CP949'

df_key1_raw = pd.read_csv(os.path.join(filepath_data, 'lung_trtm_casb.csv'), na_values='\\N', encoding=encoding)

condition1 = ['Cisplatin', 'Etoposide']
key_id_1 = df_key1_raw[df_key1_raw['CSTR_MAIN_INGR_NM'].isin(condition1)]['PT_SBST_NO'].unique()

df_key2_raw = pd.read_csv(os.path.join(filepath_data, 'lung_pt_bsnf.csv'), na_values='\\N', encoding=encoding)

condition2 = 18
df_key2 = df_key2_raw[df_key2_raw['PT_SBST_NO'].isin(key_id_1)]
key_id_2 = df_key2[df_key2['BSPT_IDGN_AGE'] >= condition2]['PT_SBST_NO'].unique()

for filename in sorted(os.listdir(filepath_data)):
    df_raw = pd.read_csv(os.path.join(filepath_data, filename), na_values='\\N', encoding=encoding)
    df = df_raw[df_raw['PT_SBST_NO'].isin(key_id_2)]

    print(filename, len(df_raw), len(df))
    df.to_csv(os.path.join(filepath_out, filename), na_rep='\\N', encoding=encoding, index=False)


#%%
# BRST
filepath_data = os.path.join('/', 'home', 'mincheol', 'ext', 'hdd1', 'data', 'CONNECT_extract', 'BRST_raw')
filepath_out = os.path.join('/', 'home', 'mincheol', 'ext', 'hdd1', 'data', 'CONNECT_extract', 'BRST')
encoding = 'CP949'

df_key1_raw = pd.read_csv(os.path.join(filepath_data, 'brst_trtm_casb.csv'), na_values='\\N', encoding=encoding)

condition1 = ['cycloPHOSphamide ', 'Doxorubicin ']
key_id_1 = df_key1_raw[df_key1_raw['ANCN_INGR_KIND_NM'].isin(condition1)]['PT_SBST_NO'].unique()

df_key2_raw = pd.read_csv(os.path.join(filepath_data, 'brst_pt_bsnf.csv'), na_values='\\N', encoding=encoding)

condition2 = 18
df_key2 = df_key2_raw[df_key2_raw['PT_SBST_NO'].isin(key_id_1)]
key_id_2 = df_key2[df_key2['IDGN_AGE'] >= condition2]['PT_SBST_NO'].unique()

for filename in sorted(os.listdir(filepath_data)):
    df_raw = pd.read_csv(os.path.join(filepath_data, filename), na_values='\\N', encoding=encoding)
    df = df_raw[df_raw['PT_SBST_NO'].isin(key_id_2)]

    print(filename, len(df_raw), len(df))
    df.to_csv(os.path.join(filepath_out, filename), na_rep='\\N', encoding=encoding, index=False)
