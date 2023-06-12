import os
par_directory = "Tensorflow/workspace/images/collected-images"
train = "Tensorflow/workspace/images/train"
test = "Tensorflow/workspace/images/test"

folders = ['aboard', 'all gone', 'baby', 'beside', 'book', 'bowl', 'bridge', 'camp', 'fond', 'friend', 'high', 'house', 'how many', 'i', 'i love you', 'marry', 'medal', 'mid day', 'middle', 'money', 'mother', 'opposite', 'rose', 'see', 'short', 'thank you', 'write', 'yes', 'you']

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