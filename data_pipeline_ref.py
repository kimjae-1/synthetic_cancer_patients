import os
import wget
import tarfile

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


# Dataset 1
# The PhysioNet Computing in Cardiology Challenge 2012 - Predicting Mortality of ICU Patients (Version: 1.0.0)
# https://www.physionet.org/content/challenge-2012/1.0.0/
def download_physionet2012(filepath=os.path.join('.', 'data', 'physionet-challenge-2012')):
    os.makedirs(filepath, exist_ok=True)

    if not os.listdir(filepath):
        wget.download('https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz', out=filepath)
        wget.download('https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz', out=filepath)
        wget.download('https://physionet.org/files/challenge-2012/1.0.0/set-c.tar.gz', out=filepath)
        wget.download('https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt', out=filepath)
        wget.download('https://physionet.org/files/challenge-2012/1.0.0/Outcomes-b.txt', out=filepath)
        wget.download('https://physionet.org/files/challenge-2012/1.0.0/Outcomes-c.txt', out=filepath)

        with tarfile.open(os.path.join(filepath, 'set-a.tar.gz'), 'r:gz') as set_a:
            set_a.extractall(filepath)

        with tarfile.open(os.path.join(filepath, 'set-b.tar.gz'), 'r:gz') as set_b:
            set_b.extractall(filepath)

        with tarfile.open(os.path.join(filepath, 'set-c.tar.gz'), 'r:gz') as set_c:
            set_c.extractall(filepath)

    else:
        pass

    return


def load_physionet2012(set_name, filepath=os.path.join('.', 'data', 'physionet-challenge-2012')):
    """
    Returns:
        data_general -- General Descriptors
            shape: (N, (x, m), d)
        data_ts -- Time Series
            shape: (N, (x, m), t, d)
        data_time -- Time Stamp
            shape: (N, t)
        data_outputs -- in-hospital mortality
            shape: (N, )
    """

    var_list = [
        # "RecordID",   # General Descriptors; key ID
        "Age",        # General Descriptors
        "Gender",     # General Descriptors; categorical (0, 1)
        "Height",     # General Descriptors
        "ICUType",    # General Descriptors; categorical (1, 2, 3, 4)
        "Albumin",
        "ALP",
        "ALT",
        "AST",
        "Bilirubin",
        "BUN",
        "Cholesterol",
        "Creatinine",
        "DiasABP",
        "FiO2",
        "GCS",
        "Glucose",
        "HCO3",
        "HCT",
        "HR",
        "K",
        "Lactate",
        "Mg",
        "MAP",
        "MechVent",   # categorical (0, 1)
        "Na",
        "NIDiasABP",
        "NIMAP",
        "NISysABP",
        "PaCO2",
        "PaO2",
        "pH",
        "Platelets",
        "RespRate",
        "SaO2",
        "SysABP",
        "Temp",
        "TroponinI",
        "TroponinT",
        "Urine",
        "WBC",
        "Weight",
    ]

    # Load inputs data
    data_inputs = []
    data_time = []

    filepath_set = os.path.join(filepath, 'set-' + set_name)
    for filename in sorted(os.listdir(filepath_set)):
        df_raw = pd.read_csv(os.path.join(filepath_set, filename), header=0,
                             parse_dates=['Time'], date_parser=(lambda x: pd.to_timedelta(x + ':00')))
        df = df_raw.pivot_table(index='Time', columns='Parameter', values='Value', aggfunc='mean')

        var_add = list(set(var_list) - set(df.columns))
        var_sub = list(set(df.columns) - set(var_list))

        df[var_add] = np.nan
        df = df.drop(var_sub, axis=1)
        df = df[var_list]

        x = df.to_numpy()
        x[x == -1] = np.nan
        data_inputs.append(x)

        t = ((df.index.days + df.index.seconds / (24 * 60 * 60)) / 2).to_numpy()
        data_time.append(t)

    # Post-zero-padding by max t_len
    max_t_len = max(map(len, data_time))

    for i, xi in enumerate(data_inputs):
        xi = np.pad(xi, pad_width=((0, max_t_len - xi.shape[0]), (0, 0)), mode='constant', constant_values=np.nan)
        data_inputs[i] = xi

    for i, xi in enumerate(data_time):
        xi = np.pad(xi, pad_width=(0, max_t_len - xi.shape[0]), mode='constant', constant_values=np.nan)
        data_time[i] = xi

    data_inputs = np.array(data_inputs, dtype=np.float32)
    data_time = np.array(data_time, dtype=np.float32)

    # Split 'General Descriptors' and 'Time Series'
    data_general = data_inputs[:, 0, :4]
    data_ts = data_inputs[:, :, 4:]

    # One-hot-encoding for 'General Descriptors' categorical variable
    cat_axis = [1, 3]
    cat_new_class = 0

    data_general[:, 3] -= 1

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

    # One-hot-encoding for 'Time Series' categorical variable
    cat_axis = 19
    cat_class = [0., 1.]

    data_ts[:, :, cat_axis][np.isnan(data_ts[:, :, cat_axis])] = cat_class[-1] + 1

    ohe = np.eye(len(cat_class) + 1)[data_ts[:, :, cat_axis].astype(np.int32)]
    for idx, value in np.ndenumerate(ohe):
        if idx[2] == cat_class[-1] + 1:
            if ohe[idx] == 1:
                ohe[idx[0], idx[1], :] = np.nan
    ohe = ohe[:, :, :-1]

    data_ts = np.insert(data_ts, obj=[cat_axis+1], values=ohe, axis=2)
    data_ts = np.delete(data_ts, obj=cat_axis, axis=2)

    # Inputs mask vector
    data_general_mask = (~ np.isnan(data_general)).astype(np.int32)
    data_ts_mask = (~ np.isnan(data_ts)).astype(np.int32)

    data_general = np.concatenate((data_general[:, np.newaxis, :], data_general_mask[:, np.newaxis, :]), axis=1)
    data_ts = np.concatenate((data_ts[:, np.newaxis, :, :], data_ts_mask[:, np.newaxis, :, :]), axis=1)

    data_general = data_general.astype(np.float32)
    data_ts = data_ts.astype(np.float32)

    # Load outputs data
    df_outputs = pd.read_csv(os.path.join(filepath, 'Outcomes-' + set_name + '.txt'), header=0)
    data_outputs = np.array(df_outputs['In-hospital_death'], dtype=np.float32)

    return data_general, data_ts, data_time, data_outputs


def physionet2012(filepath=os.path.join('.', 'data', 'physionet-challenge-2012')):
    """
    Returns:
        data -- (X_0, X_t, X_info, y)
            X_0 -- shape: (N, (x, m), d)
            X_t -- shape: (N, (x, m), t, d)
            X_info -- shape: (N, t)
            y -- shape: (N, )
    """

    download_physionet2012(filepath)

    X_0_cat_axis, X_t_cat_axis = [1, 2, 4, 5, 6, 7], [19, 20]

    X_0_train_raw, X_t_train_raw, X_info_train_raw, y_train = load_physionet2012(set_name='a', filepath=filepath)
    X_0_train, X_0_train_mean, X_0_train_std = normalization_0(X_0_train_raw.copy(), X_0_cat_axis, train=True)
    X_t_train, X_t_train_mean, X_t_train_std = normalization_t(X_t_train_raw.copy(), X_t_cat_axis, train=True)
    X_info_train = np.nan_to_num(X_info_train_raw, nan=1)

    X_0_valid_raw, X_t_valid_raw, X_info_valid_raw, y_valid = load_physionet2012(set_name='b', filepath=filepath)
    X_0_valid = normalization_0(X_0_valid_raw.copy(), X_0_cat_axis, X_0_train_mean, X_0_train_std, train=False)
    X_t_valid = normalization_t(X_t_valid_raw.copy(), X_t_cat_axis, X_t_train_mean, X_t_train_std, train=False)
    X_info_valid = np.nan_to_num(X_info_valid_raw, nan=1)

    X_0_test_raw, X_t_test_raw, X_info_test_raw, y_test = load_physionet2012(set_name='c', filepath=filepath)
    X_0_test = normalization_0(X_0_test_raw.copy(), X_0_cat_axis, X_0_train_mean, X_0_train_std, train=False)
    X_t_test = normalization_t(X_t_test_raw.copy(), X_t_cat_axis, X_t_train_mean, X_t_train_std, train=False)
    X_info_test = np.nan_to_num(X_info_test_raw, nan=1)

    data_train = (X_0_train, X_t_train, X_info_train, y_train)
    data_valid = (X_0_valid, X_t_valid, X_info_valid, y_valid)
    data_test = (X_0_test, X_t_test, X_info_test, y_test)

    return data_train, data_valid, data_test


# Dataset 2
# MIMIC-3 Clinical Database (Version: 1.4)
# https://www.physionet.org/content/mimiciii/1.4/
def download_mimic3(username, filepath=os.path.join('.', 'data', 'physionet-mimic-3')):
    """
    MIMIC-3 is credentialed access database,
    which requests approved physionet account ID and password.

    Arguments:
        username -- your own approved physionet account
    """

    os.makedirs(filepath, exist_ok=True)

    if not os.listdir(filepath):
        files = ['https://physionet.org/files/mimiciii/1.4/ADMISSIONS.csv.gz',
                 'https://physionet.org/files/mimiciii/1.4/CHARTEVENTS.csv.gz',
                 'https://physionet.org/files/mimiciii/1.4/ICUSTAYS.csv.gz',
                 'https://physionet.org/files/mimiciii/1.4/PATIENTS.csv.gz']

        os.system('wget -P {0} --user {1} --ask-password {2}'.format(filepath, username, ' '.join(files)))

        for filename in sorted(os.listdir(filepath)):
            os.system('gzip -d {}'.format(os.path.join(filepath, filename)))

    else:
        pass

    return


def load_mimic3(chunksize=10**7, filepath=os.path.join('.', 'data', 'physionet-mimic-3')):
    """
    Arguments:
        chunksize -- determined depending on RAM size
    Returns:
        data_general -- General Descriptors
            shape: (N, (x, m), d)
        data_ts -- Time Series
            shape: (N, (x, m), t, d)
        data_time -- Time Stamp
            shape: (N, t)
        data_outputs -- in-hospital mortality
            shape: (N, )
    """

    ## Cohort selection
    df_adm_raw = pd.read_csv(os.path.join(filepath, 'ADMISSIONS.csv'))
    df_icu_raw = pd.read_csv(os.path.join(filepath, 'ICUSTAYS.csv'))
    df_pt_raw = pd.read_csv(os.path.join(filepath, 'PATIENTS.csv'))

    # ICU length of stay >= 2 days (48 hours)
    df_icu = df_icu_raw[df_icu_raw['LOS'] >= 2]
    df_adm = df_adm_raw[df_adm_raw['HADM_ID'].isin(df_icu['HADM_ID'].unique())]

    cols_icu = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'FIRST_CAREUNIT', 'LAST_CAREUNIT', 'INTIME', 'OUTTIME', 'LOS']
    cols_adm = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ADMISSION_TYPE']
    df_icu_adm = pd.merge(df_icu[cols_icu], df_adm[cols_adm], how='left', on=['SUBJECT_ID', 'HADM_ID'])

    # Adult patients (age >= 16)
    cols_pt = ['SUBJECT_ID', 'GENDER', 'DOB', 'DOD']
    df_general = pd.merge(df_icu_adm, df_pt_raw[cols_pt], how='left', on=['SUBJECT_ID'])

    df_general['GENDER'] = df_general['GENDER'].map(lambda x: 1 if x == 'M' else (0 if x == 'F' else np.nan))

    cols_time = ['INTIME', 'OUTTIME', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'DOB', 'DOD']
    for col in cols_time:
        df_general[col] = pd.to_datetime(df_general[col])

    df_general['AGE'] = df_general.apply(lambda x: (x['INTIME'].year - x['DOB'].year) - ((x['INTIME'].month, x['INTIME'].day) < (x['DOB'].month, x['DOB'].day)), axis=1)
    df_general = df_general[df_general['AGE'] >= 16]

    # Select first ICU stay if multi ICU stays
    df_general = df_general.sort_values(by=['SUBJECT_ID', 'ADMITTIME', 'HADM_ID', 'INTIME', 'ICUSTAY_ID'])
    df_general = df_general[~ df_general['HADM_ID'].duplicated(keep='first')]

    # In-hospital mortality
    df_general['DEATH'] = df_general.apply(lambda x: 1 if x['DEATHTIME'] >= x['INTIME'] else (9 if x['DEATHTIME'] < x['INTIME'] else 0), axis=1)
    df_general = df_general[df_general['DEATH'] != 9]

    ## Data extraction
    var_dict = {
        # "CRR": [3348, ],
        # "CRR(R)": [115, 223951],
        # "CRR(L)": [8377, 224308],
        "DBP": [8368, 8440, 8441, 8555, 220051, 220180],
        # "EtCO2": [1817, 228640],
        "FiO2": [2981, 3420, 3422, 223835],
        "Glucose": [807, 811, 1529, 3744, 3745, 220621, 225664, 226537],
        "HR": [211, 220045],
        "pH": [780, 860, 1126, 1673, 3839, 4202, 4753, 6003, 220274, 220734, 223830, 228243],
        "RR": [615, 618, 220210, 224690],
        "SBP": [51, 442, 455, 6701, 220050, 220179],
        "SpO2": [646, 220277],
        "Temp(C)": [676, 223762],
        "Temp(F)": [678, 223761],
        "TGCS": [198, 226755, 227013],
        # "Urine": [43053, 43171, 43173, 43333, 43347, 43348, 43355, 43365, 43373, 43374, 43379, 43380, 43431, 43519,
        #           43522, 43537, 43576, 43583, 43589, 43638, 43647, 43654, 43811, 43812, 43856, 44706, 45304, 227519]
    }

    var_items = []
    for items in var_dict.values():
        var_items += items
    var_items = sorted(var_items)

    # Variable selection
    df_chr_chks = []
    cols_general = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME']
    for df_chr_raw_chk in pd.read_csv(os.path.join(filepath, 'CHARTEVENTS.csv'), chunksize=chunksize):
        df_chr_chk = df_chr_raw_chk[df_chr_raw_chk['ICUSTAY_ID'].isin(df_general['ICUSTAY_ID'].unique())]
        df_chr_chk = df_chr_chk[df_chr_chk['ITEMID'].isin(var_items)]
        df_chr_chk = df_chr_chk[df_chr_chk['VALUENUM'].notnull()]
        df_chr_chk = df_chr_chk[df_chr_chk['ERROR'] != 1]

        df_chr_chk['CHARTTIME'] = pd.to_datetime(df_chr_chk['CHARTTIME'])
        df_chr_chk = pd.merge(df_chr_chk, df_general[cols_general], how='left', on=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'])

        df_chr_chk['TIMESTAMP'] = df_chr_chk['CHARTTIME'] - df_chr_chk['INTIME']
        idx_condition_1 = df_chr_chk['TIMESTAMP'].dt.total_seconds()/(60 * 60 * 24) <= 2
        idx_condition_2 = df_chr_chk['TIMESTAMP'].dt.total_seconds()/(60 * 60 * 24) >= 0
        df_chr_chk = df_chr_chk[idx_condition_1 & idx_condition_2]

        df_chr_chks.append(df_chr_chk)
    df_chr_raw = pd.concat(df_chr_chks)
    df_general = df_general[df_general['ICUSTAY_ID'].isin(df_chr_raw['ICUSTAY_ID'].unique())]

    cols_chr = ['ICUSTAY_ID', 'ITEMID', 'VALUENUM', 'TIMESTAMP']
    df_chr = df_chr_raw[cols_chr]
    df_chr = df_chr.sort_values(by=['ICUSTAY_ID', 'TIMESTAMP'])

    # Time length truncation
    icu_id_trunc = df_chr['ICUSTAY_ID'].unique()[df_chr['TIMESTAMP'].groupby(df_chr['ICUSTAY_ID']).count() > 2000]
    df_general = df_general[~ df_general['ICUSTAY_ID'].isin(icu_id_trunc)]
    df_chr = df_chr[~ df_chr['ICUSTAY_ID'].isin(icu_id_trunc)]

    def id2name(x):
        for k, v in var_dict.items():
            if x['ITEMID'] in v:
                return k
    df_chr['ITEM'] = df_chr.apply(id2name, axis=1)

    data_general = []
    data_ts = []
    data_time = []
    data_outputs = []

    for id in sorted(df_general['ICUSTAY_ID'].unique()):
        df = df_chr[df_chr['ICUSTAY_ID'] == id]
        df = df.pivot_table(index='TIMESTAMP', columns='ITEM', values='VALUENUM', aggfunc='mean')

        var_add = list(set(var_dict.keys()) - set(df.columns))
        var_sub = list(set(df.columns) - set(var_dict.keys()))

        df[var_add] = np.nan
        df = df.drop(var_sub, axis=1)
        df = df[var_dict.keys()]

        x_ts = df.to_numpy()
        data_ts.append(x_ts)

        t = ((df.index.days + df.index.seconds / (24 * 60 * 60)) / 2).to_numpy()
        data_time.append(t)

        x_general = df_general[df_general['ICUSTAY_ID'] == id][['AGE', 'GENDER']].to_numpy()
        data_general.append(x_general)

        y = df_general[df_general['ICUSTAY_ID'] == id]['DEATH'].to_numpy()
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
    data_outputs = np.array(data_outputs, dtype=np.float32)

    # One-hot-encoding for 'GENDER' categorical variable
    cat_axis = 1
    cat_class = np.unique(data_general[:, cat_axis])

    ohe = np.eye(len(cat_class))[data_general[:, cat_axis].astype(np.int32)]

    data_general = np.insert(data_general, obj=[cat_axis + 1], values=ohe, axis=1)
    data_general = np.delete(data_general, obj=cat_axis, axis=1)

    # Inputs mask vector
    data_general_mask = (~ np.isnan(data_general)).astype(np.int32)
    data_ts_mask = (~ np.isnan(data_ts)).astype(np.int32)

    data_general = np.concatenate((data_general[:, np.newaxis, :], data_general_mask[:, np.newaxis, :]), axis=1)
    data_ts = np.concatenate((data_ts[:, np.newaxis, :, :], data_ts_mask[:, np.newaxis, :, :]), axis=1)

    data_general = data_general.astype(np.float32)
    data_ts = data_ts.astype(np.float32)

    return data_general, data_ts, data_time, data_outputs


def mimic3(username, seed=42, chunksize=10**7, filepath=os.path.join('.', 'data', 'physionet-mimic-3')):
    """
    Arguments:
        username -- your own approved physionet account
    Returns:
        data -- (X_0, X_t, X_info, y)
            X_0 -- shape: (N, (x, m), d)
            X_t -- shape: (N, (x, m), t, d)
            X_info -- shape: (N, t)
            y -- shape: (N, )
    """

    download_mimic3(username, filepath)
    X_0_raw, X_t_raw, X_info_raw, y = load_mimic3(chunksize, filepath)

    # Train:Valid:Test = 6:2:2
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    for train_idx, hold_idx in sss.split(X_0_raw, y):
        X_0_train_raw, X_t_train_raw, X_info_train_raw = X_0_raw[train_idx], X_t_raw[train_idx], X_info_raw[train_idx]
        y_train = y[train_idx]

        X_0_hold_raw, X_t_hold_raw, X_info_hold_raw = X_0_raw[hold_idx], X_t_raw[hold_idx], X_info_raw[hold_idx]
        y_hold = y[hold_idx]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    for valid_idx, test_idx in sss.split(X_0_hold_raw, y_hold):
        X_0_valid_raw, X_t_valid_raw, X_info_valid_raw = X_0_hold_raw[valid_idx], X_t_hold_raw[valid_idx], X_info_hold_raw[valid_idx]
        y_valid = y_hold[valid_idx]

        X_0_test_raw, X_t_test_raw, X_info_test_raw = X_0_hold_raw[test_idx], X_t_hold_raw[test_idx], X_info_hold_raw[test_idx]
        y_test = y_hold[test_idx]

    X_0_cat_axis, X_t_cat_axis = [1, 2], []

    X_0_train, X_0_train_mean, X_0_train_std = normalization_0(X_0_train_raw.copy(), X_0_cat_axis, train=True)
    X_t_train, X_t_train_mean, X_t_train_std = normalization_t(X_t_train_raw.copy(), X_t_cat_axis, train=True)
    X_info_train = np.nan_to_num(X_info_train_raw, nan=1)

    X_0_valid = normalization_0(X_0_valid_raw.copy(), X_0_cat_axis, X_0_train_mean, X_0_train_std, train=False)
    X_t_valid = normalization_t(X_t_valid_raw.copy(), X_t_cat_axis, X_t_train_mean, X_t_train_std, train=False)
    X_info_valid = np.nan_to_num(X_info_valid_raw, nan=1)

    X_0_test = normalization_0(X_0_test_raw.copy(), X_0_cat_axis, X_0_train_mean, X_0_train_std, train=False)
    X_t_test = normalization_t(X_t_test_raw.copy(), X_t_cat_axis, X_t_train_mean, X_t_train_std, train=False)
    X_info_test = np.nan_to_num(X_info_test_raw, nan=1)

    data_train = (X_0_train, X_t_train, X_info_train, y_train)
    data_valid = (X_0_valid, X_t_valid, X_info_valid, y_valid)
    data_test = (X_0_test, X_t_test, X_info_test, y_test)

    return data_train, data_valid, data_test


# Dataset 3
# Dataset 3 link


# General
def normalization_0(data, cat_axis, data_mean=None, data_std=None, train=True):
    """
    Returns:
        data -- Standard Normal Distribution, N(0, 1)
    """

    axis = np.arange(data.shape[-1])
    axis = np.delete(axis, cat_axis)

    if train:
        data_mean = np.nanmean(data[:, 0, axis], axis=0)
        data_std = np.nanstd(data[:, 0, axis], axis=0)

    data[:, 0, axis] = (data[:, 0, axis] - data_mean) / data_std
    data = np.nan_to_num(data, nan=0)

    if train:
        return data, data_mean, data_std
    else:
        return data


def normalization_t(data, cat_axis, data_mean=None, data_std=None, train=True):
    """
    Returns:
        data -- Standard Normal Distribution, N(0, 1)
    """

    axis = np.arange(data.shape[-1])
    axis = np.delete(axis, cat_axis)

    if train:
        data_mean = np.nanmean(data[:, 0, :, :][:, :, axis], axis=(0, 1))
        data_std = np.nanstd(data[:, 0, :, :][:, :, axis], axis=(0, 1))

    data[:, 0, :, :][:, :, axis] = (data[:, 0, :, :][:, :, axis] - data_mean) / data_std
    data = np.nan_to_num(data, nan=0)

    if train:
        return data, data_mean, data_std
    else:
        return data


def slide_window(data, window):
    """
    Arguments:
        data -- shape: (N, (x, m), t, d)

    Returns:
        data_windows -- shape: (N, t, 2 * w, d)
    """

    windows = []
    for i in range(window, 0, -1):
        slide = np.pad(array=data[:, 0, :-i, :], pad_width=((0, 0), (i, 0), (0, 0)), mode='edge')
        windows.append(slide)

    for i in range(1, window + 1):
        slide = np.pad(array=data[:, 0, i:, :], pad_width=((0, 0), (0, i), (0, 0)), mode='edge')
        windows.append(slide)

    data_windows = np.concatenate([np.expand_dims(data_window, axis=2) for data_window in windows], axis=2)

    return data_windows
