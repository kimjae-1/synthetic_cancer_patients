import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
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


def load_data(filepath=os.path.join(os.getcwd(),'data'),
              encoding='CP949', lesion='CLRC'):

    df_pt_bsnf_raw = pd.read_csv(os.path.join(filepath, lesion, lesion.lower() + '_pt_bsnf.csv'), na_values='\\N',
                                 encoding=encoding)
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
        'BUN',
        'Bilirubin, Total',
        'CA 19-9',
        'CEA',
        'CRP, Quantitative (High Sensitivity)',
        'ESR (Erythrocyte Sedimentation Rate)',
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


def clrc_diag(filepath=os.path.join(os.getcwd(),'data'),
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

def combine_general_ts(data_train, data_valid, data_test, data_normalize):
    """
    Var time: General Descriptors / Time Series
    Var type: Continuous / Categorical
    Arguments:
        data -- (X_0, X_t, X_info, y)
            X_0 -- shape: (N, (x, m), d_0)
            X_t -- shape: (N, (x, m), t, d_t)
            X_info -- shape: (N, t)
            y -- shape: (N, 1)
        normalize -- (X_0, X_t)
            X -- (min, max, cat_axis)
    Returns:
        data -- (X_0t, X_info, y)
            X_0t -- shape: (N, (x, m), 1+t, d_0t)
            X_info -- shape: (N, 1+t)
            y -- shape: (N, 1)
        normalize -- (X_0t)
            X -- (min, max, cat_axis)
    """

    # data
    X_0_train, X_t_train, X_info_train, y_train = data_train
    X_0_valid, X_t_valid, X_info_valid, y_valid = data_valid
    X_0_test, X_t_test, X_info_test, y_test = data_test

    pad_width_0_train = ((0, 0), (0, 0), (0, X_info_train.shape[-1]), (0, 0))
    pad_width_0_valid = ((0, 0), (0, 0), (0, X_info_valid.shape[-1]), (0, 0))
    pad_width_0_test = ((0, 0), (0, 0), (0, X_info_test.shape[-1]), (0, 0))

    pad_width_t = ((0, 0), (0, 0), (1, 0), (0, 0))
    pad_width_info = ((0, 0), (1, 0))

    X_0_train = np.pad(X_0_train[:, :, np.newaxis, :], pad_width=pad_width_0_train, mode='constant', constant_values=0)
    X_0_valid = np.pad(X_0_valid[:, :, np.newaxis, :], pad_width=pad_width_0_valid, mode='constant', constant_values=0)
    X_0_test = np.pad(X_0_test[:, :, np.newaxis, :], pad_width=pad_width_0_test, mode='constant', constant_values=0)

    X_t_train = np.pad(X_t_train, pad_width=pad_width_t, mode='constant', constant_values=0)
    X_t_valid = np.pad(X_t_valid, pad_width=pad_width_t, mode='constant', constant_values=0)
    X_t_test = np.pad(X_t_test, pad_width=pad_width_t, mode='constant', constant_values=0)

    X_0t_train = np.concatenate((X_0_train, X_t_train), axis=-1)
    X_0t_valid = np.concatenate((X_0_valid, X_t_valid), axis=-1)
    X_0t_test = np.concatenate((X_0_test, X_t_test), axis=-1)

    X_info_train = np.pad(X_info_train, pad_width=pad_width_info, mode='constant', constant_values=0)
    X_info_valid = np.pad(X_info_valid, pad_width=pad_width_info, mode='constant', constant_values=0)
    X_info_test = np.pad(X_info_test, pad_width=pad_width_info, mode='constant', constant_values=0)

    data_train = (X_0t_train, X_info_train, y_train)
    data_valid = (X_0t_valid, X_info_valid, y_valid)
    data_test = (X_0t_test, X_info_test, y_test)

    # normalize
    X_0_train_normalize, X_t_train_normalize = data_normalize
    X_0_train_min, X_0_train_max, X_0_cat_axis = X_0_train_normalize
    X_t_train_min, X_t_train_max, X_t_cat_axis = X_t_train_normalize

    X_train_min = np.concatenate((X_0_train_min, X_t_train_min))
    X_train_max = np.concatenate((X_0_train_max, X_t_train_max))
    X_cat_axis = X_0_cat_axis + X_t_cat_axis

    data_normalize = (X_train_min, X_train_max, X_cat_axis)

    return data_train, data_valid, data_test, data_normalize


def convert_tensor(data_train, data_valid, data_test, batch_size):
    """
    Arguments:
        data -- (X_0t, X_info, y)
            X_0t -- shape: (N, (x, m), t, d_0t)
            X_info -- shape: (N, t)
            y -- shape: (N, 1)
    Returns:
        data -- (inputs, outputs)
            inputs -- {"inputs_t", "inputs_time"}
                inputs_t -- shape: (N, (x, m), t, d_0t)
                inputs_time -- shape: (N, t)
            outputs -- {"pred", "recon"}
                pred -- shape: (N, 1)
                recon -- shape: (N, (x, m), t, d_0t)
    """

    X_0t_train, X_info_train, y_train = data_train
    X_0t_valid, X_info_valid, y_valid = data_valid
    X_0t_test, X_info_test, y_test = data_test

    X_train = {"inputs_t": X_0t_train, "inputs_time": X_info_train}
    X_valid = {"inputs_t": X_0t_valid, "inputs_time": X_info_valid}
    X_test = {"inputs_t": X_0t_test, "inputs_time": X_info_test}

    Y_train = {"pred": y_train, "recon": X_0t_train}
    Y_valid = {"pred": y_valid, "recon": X_0t_valid}
    Y_test = {"pred": y_test, "recon": X_0t_test}

    tensor_train = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    tensor_train = tensor_train.shuffle(len(y_train)).batch(batch_size)

    tensor_valid = tf.data.Dataset.from_tensor_slices((X_valid, Y_valid))
    tensor_valid = tensor_valid.batch(batch_size)

    tensor_test = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    tensor_test = tensor_test.batch(batch_size)

    return tensor_train, tensor_valid, tensor_test