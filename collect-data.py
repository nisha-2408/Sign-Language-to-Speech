import cv2
import os
import time
import uuid
cap=cv2.VideoCapture(0)
directory='Tensorflow/workspace/images/collected-images'
folders = [ 'Thank You', 'I Love You', 'Yes']
count = {}
while True:
    _,frame=cap.read()
    for i in folders:
        count[i] = len(os.listdir(directory+"/"+i))
    cv2.imshow("data",frame)
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == ord('0'):
        imagename = os.path.join(directory, folders[0], folders[0]+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imagename,frame)
    if interrupt & 0xFF == ord('1'):
        imagename = os.path.join(directory, folders[1], folders[1]+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imagename,frame)
    if interrupt & 0xFF == ord('2'):
        imagename = os.path.join(directory, folders[2], folders[2]+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imagename,frame)
    if interrupt & 0xFF == ord('a'):
        break


cap.release()
cv2.destroyAllWindows()