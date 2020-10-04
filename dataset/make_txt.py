import glob
import random
import os
import json
LENA_folder = "/ws/ifp-10_3/hasegawa/junzhez2/LENA_allday"
ages = ["03m", "06m", "09m", "12m"]
trainfiles, testfiles = [], []
for age in ages:
    wavfiles = glob.glob(os.path.join(LENA_folder, age, '*.wav'), recursive = True)
    wavfiles.sort()
    print(age, len(wavfiles), 'files') # 40 files
    num_train = int(len(wavfiles)*0.8)
    trainfiles += wavfiles[:num_train]
    testfiles += wavfiles[num_train:]
random.seed(0)
random.shuffle(trainfiles)
random.shuffle(testfiles)
with open(os.path.join(LENA_folder, 'train.txt'), 'w+') as file:
    for trainfile in trainfiles:
        file.write(trainfile)
        file.write('\n')
with open(os.path.join(LENA_folder, 'val.txt'), 'w+') as file:
    for testfile in testfiles:
        file.write(testfile)
        file.write('\n')
