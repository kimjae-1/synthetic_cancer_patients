import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Dataset 1
# CLRC CEA Time-series data generation
def cohort_selection(df_pt_bsnf):
    """
    (1) CANCER_PT_BSNF : BSPT_FRST_DIAG_YMD
    (2) CANCER_PT_BSNF : (BSPT_FRST_OPRT_YMD, BSPT_FRST_TRTM_STRT_YMD)
    0 <= Diff = MIN((2)) - (1) <= 3 months
    """

    df_frst_ymd = df_pt_bsnf[['PT_SBST_NO', 'BSPT_FRST_DIAG_YMD', 'BSPT_FRST_OPRT_YMD', 'BSPT_FRST_ANCN_TRTM_STRT_YMD',
                              'BSPT_FRST_RDT_STRT_YMD']]
    df_frst_ymd['BSPT_FRST_MIN_YMD'] = df_frst_ymd.iloc[:, 2:4].min(axis=1)
    for col in df_frst_ymd.columns[1:]:
        df_frst_ymd[col] = pd.to_datetime(df_frst_ymd[col], format='%Y%m%d')
    df_frst_ymd['BSPT_FRST_DIFF'] = df_frst_ymd['BSPT_FRST_MIN_YMD'] - df_frst_ymd['BSPT_FRST_DIAG_YMD']

    df_frst_ymd = df_frst_ymd[df_frst_ymd['BSPT_FRST_DIFF'].dt.days.notnull()]
    df_frst_ymd = df_frst_ymd[df_frst_ymd['BSPT_FRST_DIFF'].dt.days <= 90]
    df_frst_ymd = df_frst_ymd[df_frst_ymd['BSPT_FRST_DIFF'].dt.days >= 0]

    pt_key_id = sorted(df_frst_ymd['PT_SBST_NO'].unique())

    return pt_key_id


def load_data(filepath=os.path.join('/', 'home', 'mincheol', 'ext', 'hdd1', 'data', 'CONNECT', 'SMC', '202107'),
              encoding='CP949', lesion='CLRC'):

    df_pt_bsnf_raw = pd.read_csv(os.path.join(filepath, lesion, lesion.lower() + '_pt_bsnf.csv'), na_values='\\N',
                                 encoding=encoding)
    # df_oprt_nfrm_raw = pd.read_csv(os.path.join(filepath, lesion, lesion.lower() + '_oprt_nfrm.csv'), na_values='\\N',
    #                                encoding=encoding)
    # df_trtm_casb_raw = pd.read_csv(os.path.join(filepath, lesion, lesion.lower() + '_trtm_casb.csv'), na_values='\\N',
    #                                encoding=encoding)
    df_ex_diag1_raw = pd.read_csv(os.path.join(filepath, lesion, lesion.lower() + '_ex_diag1.csv'), na_values='\\N',
                                  encoding=encoding)
    df_ex_diag2_raw = pd.read_csv(os.path.join(filepath, lesion, lesion.lower() + '_ex_diag2.csv'), na_values='\\N',
                                  encoding=encoding)

    ## Cohort selection
    pt_key_id = cohort_selection(df_pt_bsnf_raw)

    ## Data extraction
    # df_pt_bsnf
    df_pt_bsnf = df_pt_bsnf_raw[df_pt_bsnf_raw['PT_SBST_NO'].isin(pt_key_id)]
    df_pt_bsnf = df_pt_bsnf[['PT_SBST_NO', 'BSPT_IDGN_AGE', 'BSPT_SEX_CD', 'BSPT_FRST_DIAG_CD', 'BSPT_FRST_DIAG_YMD',
                             'BSPT_DEAD_YMD']]

    df_pt_bsnf['BSPT_SEX_CD'] = df_pt_bsnf['BSPT_SEX_CD'].replace({'F': 0, 'M': 1})

    diag_cd = sorted(df_pt_bsnf['BSPT_FRST_DIAG_CD'].unique())
    diag_cd = {cd: i for i, cd in enumerate(diag_cd)}
    df_pt_bsnf['BSPT_FRST_DIAG_CD'] = df_pt_bsnf['BSPT_FRST_DIAG_CD'].replace(diag_cd)

    df_pt_bsnf['BSPT_DEAD'] = df_pt_bsnf['BSPT_DEAD_YMD'].notnull().astype(np.int32)

    # # df_oprt_nfrm
    # df_oprt_nfrm = df_oprt_nfrm_raw[df_oprt_nfrm_raw['PT_SBST_NO'].isin(pt_key_id)]
    # df_oprt_nfrm = pd.DataFrame(pt_key_id, columns=['PT_SBST_NO']).merge(df_oprt_nfrm, how='outer', on='PT_SBST_NO')
    # df_oprt_nfrm = df_oprt_nfrm[['PT_SBST_NO', 'OPRT_YMD']].sort_values(by=['PT_SBST_NO', 'OPRT_YMD'])
    # df_oprt_nfrm = df_oprt_nfrm[~ df_oprt_nfrm.duplicated()]
    # df_oprt_nfrm = df_oprt_nfrm.groupby('PT_SBST_NO')['OPRT_YMD'].apply(list)

    # # df_trtm_casb
    # df_trtm_casb = df_trtm_casb_raw[df_trtm_casb_raw['PT_SBST_NO'].isin(pt_key_id)]
    # df_trtm_casb = pd.DataFrame(pt_key_id, columns=['PT_SBST_NO']).merge(df_trtm_casb, how='outer', on='PT_SBST_NO')
    # df_trtm_casb = df_trtm_casb[['PT_SBST_NO', 'CSTR_STRT_YMD', 'CSTR_END_YMD']].sort_values(by=['PT_SBST_NO', 'CSTR_STRT_YMD', 'CSTR_END_YMD'])
    # df_trtm_casb = df_trtm_casb[~ df_trtm_casb.duplicated()]
    # df_trtm_casb_strt = df_trtm_casb.groupby('PT_SBST_NO')['CSTR_STRT_YMD'].apply(list)
    # df_trtm_casb_end = df_trtm_casb.groupby('PT_SBST_NO')['CSTR_END_YMD'].apply(list)
    # df_trtm_casb = pd.merge(df_trtm_casb_strt, df_trtm_casb_end, how='left', on='PT_SBST_NO')

    # df_ex_diag
    df_ex_diag1 = df_ex_diag1_raw[df_ex_diag1_raw['PT_SBST_NO'].isin(pt_key_id)]
    df_ex_diag1 = df_ex_diag1[['PT_SBST_NO', 'CEXM_YMD', 'CEXM_NM', 'CEXM_RSLT_CONT', 'CEXM_RSLT_UNIT_CONT']]
    df_ex_diag2 = df_ex_diag2_raw[df_ex_diag2_raw['PT_SBST_NO'].isin(pt_key_id)]
    df_ex_diag2 = df_ex_diag2[['PT_SBST_NO', 'CEXM_YMD', 'CEXM_NM', 'CEXM_RSLT_CONT', 'CEXM_RSLT_UNIT_CONT']]
    df_ex_diag = pd.concat([df_ex_diag1, df_ex_diag2], axis=0, ignore_index=True).sort_values(by=['PT_SBST_NO', 'CEXM_YMD', 'CEXM_NM'])
    df_ex_diag = df_ex_diag.reset_index(drop=True)

    var_list = [
        'ALP',
        'ALT',
        'AST',
        'Albumin',
        # 'Anti-HBs Antibody',
        # 'Anti-HCV Antibody',
        # 'Anti-HIV combo',
        'BUN',
        'Bilirubin, Total',
        'CA 19-9',
        'CEA',
        'CRP, Quantitative (High Sensitivity)',
        'ESR (Erythrocyte Sedimentation Rate)',
        # 'HBsAg',
        'Protein, Total',
    ]

    # (tmp) Numeric extraction
    exclusion = ['Anti-HBs Antibody', 'Anti-HCV Antibody', 'Anti-HIV combo', 'HBsAg']
    df_ex_diag = df_ex_diag[~ df_ex_diag['CEXM_NM'].isin(exclusion)]

    df_ex_diag = pd.merge(df_ex_diag, df_pt_bsnf[['PT_SBST_NO', 'BSPT_FRST_DIAG_YMD']], how='left', on='PT_SBST_NO')
    for col in ['CEXM_YMD', 'BSPT_FRST_DIAG_YMD']:
        df_ex_diag[col] = pd.to_datetime(df_ex_diag[col], format='%Y%m%d')
    df_ex_diag['TIMESTAMP'] = df_ex_diag['CEXM_YMD'] - df_ex_diag['BSPT_FRST_DIAG_YMD']
    time_condition_1 = df_ex_diag['TIMESTAMP'].dt.days / 365 <= 5
    time_condition_2 = df_ex_diag['TIMESTAMP'].dt.days / 365 >= 0
    df_ex_diag = df_ex_diag[time_condition_1 & time_condition_2]
    df_pt_bsnf = df_pt_bsnf[df_pt_bsnf['PT_SBST_NO'].isin(df_ex_diag['PT_SBST_NO'].unique())]

    df_ex_diag['CEXM_RSLT_CONT'] = df_ex_diag['CEXM_RSLT_CONT'].astype(np.float32)

    cols_ex_diag = ['PT_SBST_NO', 'CEXM_NM', 'CEXM_RSLT_CONT', 'TIMESTAMP']
    df_ex_diag = df_ex_diag[cols_ex_diag]

    ## Convert data to tensor
    data_general = []
    data_ts = []
    data_time = []
    data_outputs = []

    for id in sorted(df_pt_bsnf['PT_SBST_NO'].unique()):
        df = df_ex_diag[df_ex_diag['PT_SBST_NO'] == id]
        df = df.pivot_table(index='TIMESTAMP', columns='CEXM_NM', values='CEXM_RSLT_CONT', aggfunc='mean')

        var_add = list(set(var_list) - set(df.columns))
        var_sub = list(set() - set(var_list))

        df[var_add] = np.nan
        df - df.drop(var_sub, axis=1)
        df = df[var_list]

        x_ts = df.to_numpy()
        data_ts.append(x_ts)

        t = (df.index.days / 365).to_numpy()
        data_time.append(t)

        cols_general = ['BSPT_IDGN_AGE', 'BSPT_SEX_CD', 'BSPT_FRST_DIAG_CD']
        x_general = df_pt_bsnf[df_pt_bsnf['PT_SBST_NO'] == id][cols_general].to_numpy()
        data_general.append(x_general)

        y = df_pt_bsnf[df_pt_bsnf['PT_SBST_NO'] == id]['BSPT_DEAD'].to_numpy()
        data_outputs.append(y)

    # Post-zero-padding by max t_len
    max_t_len = max(map(len, data_time))

    for i, xi in enumerate(data_ts):
        xi = np.pad(xi, pad_width=((0, max_t_len - xi.shape[0]), (0, 0)), mode='constant', constant_values=np.nan)
        data_ts[i] = xi

    for i, xi in enumerate(data_time):
        xi = np.pad(xi, pad_width=(0, max_t_len - xi.shape[0]), mode='constant', constant_values=np.nan)
        data_time[i] = xi

    data_general = np.squeeze(np.array(data_general, dtype=np.float32))
    data_ts = np.array(data_ts, dtype=np.float32)
    data_time = np.array(data_time, dtype=np.float32)
    data_outputs = np.array(data_outputs, dtype=np.int32)

    # One-hot-encoding for 'General Descriptors' categorical variable
    cat_axis = [1, 2]
    cat_new_class = 0

    for i in cat_axis:
        cat_class = np.unique(data_general[:, i+cat_new_class])
        cat_class = cat_class[~ np.isnan(cat_class)]

        data_general[np.isnan(data_general[:, i+cat_new_class]), i+cat_new_class] = cat_class[-1] + 1

        ohe = np.eye(len(cat_class) + 1)[data_general[:, i+cat_new_class].astype(np.int32)]
        ohe[ohe[:, -1] == 1] = np.nan
        ohe = ohe[:, :-1]

        data_general = np.insert(data_general, obj=[i+cat_new_class+1], values=ohe, axis=1)
        data_general = np.delete(data_general, obj=i+cat_new_class, axis=1)

        cat_new_class += len(cat_class) - 1

    # Inputs mask vector
    data_general_mask = (~ np.isnan(data_general)).astype(np.int32)
    data_ts_mask = (~ np.isnan(data_ts)).astype(np.int32)

    data_general = np.concatenate((data_general[:, np.newaxis, :], data_general_mask[:, np.newaxis, :]), axis=1)
    data_ts = np.concatenate((data_ts[:, np.newaxis, :, :], data_ts_mask[:, np.newaxis, :, :]), axis=1)

    data_general = data_general.astype(np.float32)
    data_ts = data_ts.astype(np.float32)

    return data_general, data_ts, data_time, data_outputs


def clrc_diag(filepath=os.path.join('/', 'home', 'mincheol', 'ext', 'hdd1', 'data', 'CONNECT', 'SMC', '202107'),
              encoding='CP949', seed=42):

    X_0_raw, X_t_raw, X_info_raw, y = load_data(filepath=filepath, encoding=encoding, lesion='CLRC')

    # Train:Valid:Test = 6:2:2
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    for train_idx, hold_idx in sss.split(X_0_raw, y):
        X_0_train_raw, X_t_train_raw, X_info_train_raw = X_0_raw[train_idx], X_t_raw[train_idx], X_info_raw[train_idx]
        y_train = y[train_idx]

        X_0_hold_raw, X_t_hold_raw, X_info_hold_raw = X_0_raw[hold_idx], X_t_raw[hold_idx], X_info_raw[hold_idx]
        y_hold = y[hold_idx]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    for valid_idx, test_idx in sss.split(X_0_hold_raw, y_hold):
        X_0_valid_raw, X_t_valid_raw, X_info_valid_raw = X_0_hold_raw[valid_idx], X_t_hold_raw[valid_idx], \
                                                         X_info_hold_raw[valid_idx]
        y_valid = y_hold[valid_idx]

        X_0_test_raw, X_t_test_raw, X_info_test_raw = X_0_hold_raw[test_idx], X_t_hold_raw[test_idx], X_info_hold_raw[
            test_idx]
        y_test = y_hold[test_idx]

    X_0_cat_axis, X_t_cat_axis = list(range(X_0_raw.shape[-1]))[1:], []

    X_0_train, X_0_train_min, X_0_train_max = normalization_0(X_0_train_raw.copy(), X_0_cat_axis, train=True)
    X_t_train, X_t_train_min, X_t_train_max = normalization_t(X_t_train_raw.copy(), X_t_cat_axis, train=True)
    X_info_train = np.nan_to_num(X_info_train_raw, nan=5)

    X_0_valid = normalization_0(X_0_valid_raw.copy(), X_0_cat_axis, X_0_train_min, X_0_train_max, train=False)
    X_t_valid = normalization_t(X_t_valid_raw.copy(), X_t_cat_axis, X_t_train_min, X_t_train_max, train=False)
    X_info_valid = np.nan_to_num(X_info_valid_raw, nan=5)

    X_0_test = normalization_0(X_0_test_raw.copy(), X_0_cat_axis, X_0_train_min, X_0_train_max, train=False)
    X_t_test = normalization_t(X_t_test_raw.copy(), X_t_cat_axis, X_t_train_min, X_t_train_max, train=False)
    X_info_test = np.nan_to_num(X_info_test_raw, nan=5)

    data_train = (X_0_train, X_t_train, X_info_train, y_train)
    data_valid = (X_0_valid, X_t_valid, X_info_valid, y_valid)
    data_test = (X_0_test, X_t_test, X_info_test, y_test)

    X_0_train_normalize = (X_0_train_min, X_0_train_max, X_0_cat_axis)
    X_t_train_normalize = (X_t_train_min, X_t_train_max, X_t_cat_axis)
    data_normalize = (X_0_train_normalize, X_t_train_normalize)

    return data_train, data_valid, data_test, data_normalize


def normalization_0(data, cat_axis, data_min=None, data_max=None, train=True):
    """
    Returns:
        data -- Normalization [0, 1]
    """

    axis = np.arange(data.shape[-1])
    axis = np.delete(axis, cat_axis)

    if train:
        data_min = np.nanmin(data[:, 0, axis], axis=0)
        data_max = np.nanmax(data[:, 0, axis], axis=0)

    data[:, 0, axis] = (data[:, 0, axis] - data_min) / (data_max - data_min)
    data = np.nan_to_num(data, nan=0)

    if train:
        return data, data_min, data_max
    else:
        return data


def normalization_t(data, cat_axis, data_min=None, data_max=None, train=True):
    """
    Returns:
        data -- Normalization [0, 1]
    """

    axis = np.arange(data.shape[-1])
    axis = np.delete(axis, cat_axis)

    if train:
        data_min = np.nanmin(data[:, 0, :, :][:, :, axis], axis=(0, 1))
        data_max = np.nanmax(data[:, 0, :, :][:, :, axis], axis=(0, 1))

    data[:, 0, :, :][:, :, axis] = (data[:, 0, :, :][:, :, axis] - data_min) / (data_max - data_min)
    data = np.nan_to_num(data, nan=0)

    if train:
        return data, data_min, data_max
    else:
        return data
