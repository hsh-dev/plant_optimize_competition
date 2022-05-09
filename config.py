CFG = {
    'IMG_SIZE': 128,
    'EPOCHS': 300,
    'LEARNING_RATE': 5e-4,
    'MIN_LEARNING_RATE': 1e-4,
    'BATCH_SIZE': 64,
    'VALID_BATCH_SIZE': 1,
    'TEST_BATCH_SIZE' : 1,
    'SEED': 41,
    'trainpath': './dataset/train',
    'testpath': './dataset/test',
    'feature': ['내부온도관측치', '외부온도관측치', '내부습도관측치', '외부습도관측치', 'CO2관측치', 'EC관측치', 
                '최근분무량', '화이트 LED동작강도', '레드 LED동작강도', '블루 LED동작강도', 
                '냉방온도', '냉방부하', '난방온도', '총추정광량', '백색광추정광량', '적색광추정광량', '청색광추정광량'],
    'empty_data': ['CASE08', 'CASE09', 'CASE22','CASE23', 'CASE26', 'CASE30', 'CASE31', 'CASE49', 'CASE59', 'CASE71', 'CASE72','CASE73'],
    'empty_data_subject': ['CASE60_20', 'CASE60_29', 'CASE60_33', 'CASE60_32', 'CASE60_26', 'CASE60_23', 
                           'CASE52_01', 'CASE02_10', 'CASE60_29', 'CASE70_23', 'CASE02_11', 'CASE34_01', 
                           'CASE56_01', 'CASE60_21', 'CASE60_27', 'CASE70_24', 'CASE70_20', 'CASE60_34',
                           'CASE60_28', 'CASE70_21', 'CASE40_01', 'CASE60_31', 'CASE63_01', 'CASE70_22',
                           'CASE60_22', 'CASE60_30', 'CASE60_24', 'CASE70_19', 'CASE60_25', 'CASE44_01']
}