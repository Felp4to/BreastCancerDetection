import os

ROOT = '../../../Datasets/CBIS_DDSM_nuovo/manifest-ZkhPvrLo5216730872708713142'
CBIS_DDSM = os.path.join(ROOT, 'CBIS-DDSM')
META = os.path.join(ROOT, 'meta.csv')
METADATA = os.path.join(ROOT, 'metadata.csv')
DICOM_INFO = os.path.join(ROOT, 'dicom_info.csv')
MASS_CASE_DESCRIPTION_TRAIN_SET = os.path.join(ROOT, 'mass_case_description_train_set.csv')
MASS_CASE_DESCRIPTION_TEST_SET = os.path.join(ROOT, 'mass_case_description_test_set.csv')
CALC_CASE_DESCRIPTION_TRAIN_SET = os.path.join(ROOT, 'calc_case_description_train_set.csv')
CALC_CASE_DESCRIPTION_TEST_SET = os.path.join(ROOT, 'calc_case_description_test_set.csv')

PATH_DICOM_DATA = '../../dataset/CBIS-DDSM/csv/dicom_info.csv'
PATH_CALC_CASE_DF = '../../dataset/CBIS-DDSM/csv/calc_case_description_train_set.csv'
PATH_MASS_CASE_DF = '../../dataset/CBIS-DDSM/csv/mass_case_description_train_set.csv'
PATH_IMAGES = '../../dataset/CBIS-DDSM/jpeg'
PATH_PNGS = '../../dataset/Breast Histopathology Images/IDC_regular_ps50_idx5/**/*.png'
PATH_PARTITIONS_ROOT = '../../partitions/'
