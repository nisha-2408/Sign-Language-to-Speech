import os

par_directory = "Tensorflow/workspace/images/collected-images"

folders = ['Thank You', 'I Love You', 'Yes']

for i in folders:
    folder = os.path.join(par_directory, i.lower())
    os.mkdir(folder)
    
