import os
par_directory = "Tensorflow/workspace/images/collected-images"
train = "Tensorflow/workspace/images/train"
test = "Tensorflow/workspace/images/test"

folders = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

for i in folders:
    folder = os.path.join(par_directory, i.lower())
    files = os.listdir(folder)
    for j in range(0, 32):
        old_folder = os.path.join(folder, files[j])
        new_folder = os.path.join(train, files[j])
        os.rename(old_folder, new_folder)
    for j in range(32, 40):
        old_folder = os.path.join(folder, files[j])
        new_folder = os.path.join(test, files[j])
        os.rename(old_folder, new_folder)
        #print(files[j])