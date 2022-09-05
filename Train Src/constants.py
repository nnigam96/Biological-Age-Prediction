# Data
DATA_PATH = "data/"
CT_PATH = DATA_PATH + 'Auto CT Data.xlsx'
OUTCOMES_PATH = DATA_PATH + 'Clinical Outcomes.xlsx'
ALL_PATH = DATA_PATH + 'OppScrData.xlsx'
NUM_CT = 11

MODEL_PATH = "models/"

BMI_COL = 4
DEATH_COL = 0
AGE_COL = 1

# data splits
TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0
TRAIN_SPLIT = 1 - TEST_SPLIT - VALIDATION_SPLIT

# LINEAR regression
linear_batch_size = 128
linear_training_epoch = 10
# epochs = LINEAR_TRAIN_EPOCHS / (len(train_dataset) / batch_size)

linear_lr_rate = 3e-3
linear_input_dim = 24
linear_output_dim = 1
linear_momentum = 0.9
linear_predict = [BMI_COL]

part_2_linear_predict = [DEATH_COL]

# Perceptron
PERCEPTRON_TRAIN_EPOCHS = 100
PERCEPTRON_TRAIN_BATCH_SIZE = 256
PERCEPTRON_TEST_BATCH_SIZE = 256

CATEGORICAL = ['bmi30', 'sex', 'tobacco', 'alcoholAbuse',  'metSx']


CATEGORICAL_OUTCOMES = ['cvdDx',  'heartFailureDx',
                        'miDx',
                        'type2DiabetesDx',
                        'femoralNeckFractureDx',
                        'unspecFemoralFractureDx',
                        'forearmFractureDx',
                        'humerusFractureDx',
                        'pathologicFractureDx',
                        'alzheimersDx',  'primaryCancerSite',
                        'primaryCancerSite2']

NAN_CT_COLUMNS = ['l1HuBmd', 'tatAreaCm2', 'totalBodyAreaEaCm2', 'vatAreaCm2',
                  'satAreaCm2', 'vatSatRatio', 'muscleHu', 'muscleAreaCm2', 'l3SmiCm2M2',
                  'aoCaAgatston', 'liverHuMedian']


NAN_CLINCAL_COLUMNS = ['clinicalFUIntervalDFromCt', 'bmi',
                       'bmi30', 'sex', 'ageAtCt', 'tobacco', 'frs10YearRisk%',
                       'frax10yFxProbOrangeWDxa', 'frax10yHipFxProbOrangeWDxa']
