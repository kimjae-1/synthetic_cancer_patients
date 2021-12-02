# import os
#
# import numpy as np
# import pandas as pd
#
# from matplotlib import pyplot as plt
#
#
# def cohort_selection(filepath_data=os.path.join('/', 'home', 'mincheol', 'ext', 'hdd1', 'data', 'CONNECT', 'SMC', '202107'),
#                      encoding='CP949'):
#
#     df_clrc_pt_bsnf = pd.read_csv(os.path.join(filepath_data, 'CLRC', 'clrc_pt_bsnf.csv'), na_values='\\N',
#                                   encoding=encoding)
#     df_clrc_pt_hlnf = pd.read_csv(os.path.join(filepath_data, 'CLRC', 'clrc_pt_hlnf.csv'), na_values='\\N',
#                                   encoding=encoding)
#     df_lung_pt_bsnf = pd.read_csv(os.path.join(filepath_data, 'LUNG', 'lung_pt_bsnf.csv'), na_values='\\N',
#                                   encoding=encoding)
#     df_lung_pt_hlnf = pd.read_csv(os.path.join(filepath_data, 'LUNG', 'lung_pt_hlnf.csv'), na_values='\\N',
#                                   encoding=encoding)
#     df_cancer_rgst = pd.read_csv(os.path.join(filepath_data, 'cancer_rgst', 'cancer_rgst.csv'), na_values='\\N',
#                                  encoding=encoding)
#
#     ###
#     # DEFINITION 01
#     # (1) CANCER_PT_BSNF, BSPT_FRST_DIAG_YMD
#     # (2) CANCER_RGST, FDX
#     # Diff = (2) - (1)
#     # if Diff >= 0:
#     #   plot distribution
#     #   select cutoff
#     # else:
#     #   delete
#     ###
#
#     df_clrc_rgst = pd.merge(df_clrc_pt_bsnf[['PT_SBST_NO', 'BSPT_FRST_DIAG_YMD']],
#                             df_cancer_rgst[['PTNO', 'FDX']],
#                             left_on='PT_SBST_NO', right_on='PTNO')
#
#     df_lung_rgst = pd.merge(df_lung_pt_bsnf[['PT_SBST_NO', 'BSPT_FRST_DIAG_YMD']],
#                             df_cancer_rgst[['PTNO', 'FDX']],
#                             left_on='PT_SBST_NO', right_on='PTNO')
#
#     (df_clrc_rgst['BSPT_FRST_DIAG_YMD'] != df_clrc_rgst['FDX']).sum()
#     (df_lung_rgst['BSPT_FRST_DIAG_YMD'] != df_lung_rgst['FDX']).sum()
#     (df_lung_rgst['BSPT_FRST_DIAG_YMD'] > df_lung_rgst['FDX']).sum()
#     (df_lung_rgst['BSPT_FRST_DIAG_YMD'] < df_lung_rgst['FDX']).sum()
#
#     ###
#     # DEFINITION 02-1
#     # (1) CANCER_PT_BSNF, BSPT_FRST_DIAG_YMD
#     # (2) CANCER_PT_BSNF, (BSPT_FRST_OPRT_YMD, BSPT_FRST_TRTM_STRT_YMD, BSPT_FRST_RDT_STRT_YMD)
#     # Diff = min((2)) - (1)
#     # if Diff >= 0:
#     #   plt distribution
#     #   select cutoff
#     # else:
#     #   delete
#     ###
#
#     df_clrc_frst_ymd = df_clrc_pt_bsnf[
#         ['PT_SBST_NO', 'BSPT_FRST_DIAG_YMD', 'BSPT_FRST_OPRT_YMD', 'BSPT_FRST_ANCN_TRTM_STRT_YMD',
#          'BSPT_FRST_RDT_STRT_YMD']]
#     df_clrc_frst_ymd['BSPT_FRST_MIN_YMD'] = df_clrc_frst_ymd.iloc[:, 2:].min(axis=1)
#     for col in df_clrc_frst_ymd.columns[1:]:
#         df_clrc_frst_ymd[col] = pd.to_datetime(df_clrc_frst_ymd[col], format='%Y%m%d')
#     df_clrc_frst_ymd['BSPT_FRST_DIFF'] = df_clrc_frst_ymd['BSPT_FRST_MIN_YMD'] - df_clrc_frst_ymd['BSPT_FRST_DIAG_YMD']
#     df_clrc_frst_ymd['BSPT_FRST_DIFF'].dt.days.hist(bins=np.arange(-360, 360, 30))
#     plt.show()
#
#     (df_clrc_frst_ymd['BSPT_FRST_DIFF'].dt.days > 365).sum()
#     (df_clrc_frst_ymd['BSPT_FRST_DIFF'].dt.days < 0).sum()
#     df_clrc_frst_ymd['BSPT_FRST_DIFF'].dt.days.isnull().sum()
#
#     """
#     Total : 30,527
#     0 <= Diff <= 365 : 17,321
#     NULL : 11,982
#     Diff > 365 : 1,090
#     Diff < 0 : 134
#     """
#
#     ###
#     # DEFINITION 02-2
#     # (1) CANCER_PT_BSNF, BSPT_FRST_DIAG_YMD
#     # (2) CANCER_PT_BSNF, (BSPT_FRST_OPRT_YMD, BSPT_FRST_TRTM_STRT_YMD)
#     # Diff = min((2)) - (1)
#     # if Diff >= 0:
#     #   plt distribution
#     #   select cutoff
#     # else:
#     #   delete
#     ###
#
#     df_clrc_frst_ymd = df_clrc_pt_bsnf[
#         ['PT_SBST_NO', 'BSPT_FRST_DIAG_YMD', 'BSPT_FRST_OPRT_YMD', 'BSPT_FRST_ANCN_TRTM_STRT_YMD',
#          'BSPT_FRST_RDT_STRT_YMD']]
#     df_clrc_frst_ymd['BSPT_FRST_MIN_YMD'] = df_clrc_frst_ymd.iloc[:, 2:4].min(axis=1)
#     for col in df_clrc_frst_ymd.columns[1:]:
#         df_clrc_frst_ymd[col] = pd.to_datetime(df_clrc_frst_ymd[col], format='%Y%m%d')
#     df_clrc_frst_ymd['BSPT_FRST_DIFF'] = df_clrc_frst_ymd['BSPT_FRST_MIN_YMD'] - df_clrc_frst_ymd['BSPT_FRST_DIAG_YMD']
#
#     (df_clrc_frst_ymd['BSPT_FRST_DIFF'].dt.days > 365).sum()
#     (df_clrc_frst_ymd['BSPT_FRST_DIFF'].dt.days < 0).sum()
#     df_clrc_frst_ymd['BSPT_FRST_DIFF'].dt.days.isnull().sum()
#
#     """
#     Total: 30,527
#     0 <= Diff <= 365 : 17,192
#     NULL : 12,383
#     Diff > 365 : 928
#     Diff < 0 : 14
#     """
#
#     ###
#     # DEFINITION 03
#     # (1) CANCER_PT_BSNF, BSPT_FRST_DIAG_YMD
#     # (2) CANCER_PT_HLNF, HLPT_ADM_YMD => 최초내원일?
#     # Diff = (1) - min((2))
#     # if Diff >= 0:
#     #   plot distribution
#     #   select cutoff
#     # else:
#     #   delete
#     ###
#
#     df_clrc_info = pd.merge(df_clrc_pt_bsnf[['PT_SBST_NO', 'BSPT_FRST_DIAG_YMD']],
#                             df_clrc_pt_hlnf[['PT_SBST_NO', 'HLPT_RCRD_YMD', 'HLPT_ADM_YMD', 'HLPT_HLNF_SEQ']],
#                             how='right', on='PT_SBST_NO')
#
#     return
