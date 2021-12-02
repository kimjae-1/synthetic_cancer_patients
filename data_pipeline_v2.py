"""
General data generation
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


def cohort_selection(df_pt_bsnf):
    """
    (1) CANCER_PT_BSNF : BSPT_FRST_DIAG_YMD
    (2) CANCER_PT_BSNF : (BSPT_FRST_OPRT_YMD, BSPT_FRST_TRTM_STRT_YMD)
    0 <= Diff = MIN((2)) - (1) <= 3 months
    """

    df_frst_ymd = df_pt_bsnf[['PT_SBST_NO', 'BSPT_FRST_DIAG_YMD', 'BSPT_FRST_OPRT_YMD', 'BSPT_FRST_ANCN_TRTM_STRT_YMD',
                              'BSPT_FRST_RDT_STRT_YMD']]

    # (BSPT_FRST_OPRT_YMD, BSPT_FRST_TRTM_STRT_YMD)
    df_frst_ymd['BSPT_FRST_MIN_YMD'] = df_frst_ymd.iloc[:, 2:4].min(axis=1)
    for col in df_frst_ymd.columns[1:]:
        df_frst_ymd[col] = pd.to_datetime(df_frst_ymd[col], format='%Y%m%d')
    df_frst_ymd['BSPT_FRST_DIFF'] = df_frst_ymd['BSPT_FRST_MIN_YMD'] - df_frst_ymd['BSPT_FRST_DIAG_YMD']

    df_frst_ymd = df_frst_ymd[df_frst_ymd['BSPT_FRST_DIFF'].dt.days.notnull()]
    df_frst_ymd = df_frst_ymd[df_frst_ymd['BSPT_FRST_DIFF'].dt.days <= 90]
    df_frst_ymd = df_frst_ymd[df_frst_ymd['BSPT_FRST_DIFF'].dt.days >= 0]

    pt_key_id = sorted(df_frst_ymd['PT_SBST_NO'].unique())

    return pt_key_id


def load_data(filepath_data=os.path.join('/', 'home', 'mincheol', 'ext', 'hdd1', 'data', 'CONNECT', 'SMC', '202107'),
              lesion='CLRC', encoding='CP949'):

    df_pt_bsnf_raw = pd.read_csv(os.path.join(filepath_data, lesion, lesion.lower() + '_pt_bsnf.csv'), na_values='\\N',
                                 encoding=encoding)
    df_pth_srgc_raw = pd.read_csv(os.path.join(filepath_data, lesion, lesion.lower() + '_pth_srgc.csv'), na_values='\\N',
                                  encoding=encoding)

    pt_key_id = cohort_selection(df_pt_bsnf_raw)
    df_pth_srgc = df_pth_srgc_raw[df_pth_srgc_raw['PT_SBST_NO'].isin(pt_key_id)]

    cols = [
        'SGPT_CELL_DIFF_CD',
        'SGPT_PATL_STAG_VL',
        'SGPT_PATL_T_STAG_VL',
        'SGPT_PATL_N_STAG_VL',
        'SGPT_PATL_M_STAG_VL',
        'SGPT_SRMV_LN_CNT',
        'SGPT_MTST_LN_CNT',
        'SGPT_SRMG_PCTS_STAT_CD',
        'SGPT_SRMG_PROX_CNCR_TXSZ_VL',
        'SGPT_SRMG_DCTS_STAT_CD',
        'SGPT_SRMG_DSTL_CNCR_TXSZ_VL',
        'SGPT_SRMG_RCTS_STAT_CD',
        'SGPT_SRMG_RADI_CNCR_TXSZ_VL',
        'SGPT_NERV_PREX_CD',
        'SGPT_VNIN_CD',
        'SGPT_ANIN_CD',
        'SGPT_TUMR_BUDD_CD',
    ]

    df_pth_srgc = df_pth_srgc[cols]

    # Delete error
    ## SGPT_MTST_LN_CNT
    df_pth_srgc = df_pth_srgc.drop(df_pth_srgc[df_pth_srgc['SGPT_MTST_LN_CNT'] == 'c,  0'].index)

    ## (SGPT_SRMG_PROX_CNCR_TXSZ_VL, SGPT_SRMG_DSTL_CNCR_TXSZ_VL, SGPT_SRMG_RADI_CNCR_TXSZ_VL)
