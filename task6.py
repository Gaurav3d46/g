import cv2
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join

# code that train the model
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return None

    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

cap = cv2.VideoCapture(0)
count = 0

while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        file_name_path = 'faces/user/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)

    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 100:
        break

cv2.destroyAllWindows()
cap.release()
print("Collecting Samples Complete")





import cv2
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join

data_path = './faces/user/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)
model  = cv2.face_LBPHFaceRecognizer.create()
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained successfully")




from smtplib import SMTP_SSL
from email.message import EmailMessage
import pywhatkit
import os, json, time, subprocess, cv2

my_server = SMTP_SSL(host='smtp.gmail.com')

msg = 'Mail from task'
def send_email():
    msg = EmailMessage()
    msg['Subject'] = 'Alert'
    msg['From'] ='xyz@gmail.com'
    msg['To'] = 'xyz@gmail.com'
    msg.set_content('Hello u loggined')
    my_server.login('xyz@gmail.com', 'Password')
    my_server.send_message(msg)


def send_whatsapp():
    from datetime import datetime
    now = datetime.now()
    current_hour = now.strftime("%H")
    current_min=now.strftime("%M")
    pywhatkit.sendwhatmsg('+91xxxxxxxxxx','Hii:-)',int(current_hour),int(current_min)+1,5)

def create_ec2():

    import boto3
    client = boto3.client('ec2')
    Key_Pair = client.create_key_pair(
        KeyName='___',
    )
    instance = client.run_instances(
        ImageId='_______',
        InstanceType='t2.micro',
        KeyName='___',
        MaxCount=1,
        MinCount=1,
        SecurityGroupIds=[
            '_____',
        ],
        SecurityGroups=[
            '_____',
        ],
    )
    EBS_Volume = client.create_volume(
        AvailabilityZone='_____',
        Size=5,
    )
    Attach_Volume = client.attach_volume(
        Device='_____',
        InstanceId='____',
        VolumeId='____',
    )

def face_detector(img, size=0.5):
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


# Open Webcam
cap = cv2.VideoCapture(0)

while True:


    _, frame = cap.read()
    image, face = face_detector(frame)

    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    results = model.predict(face)

    if results[1] < 500:
        confidence = int(100 * (1 - (results[1])/400))

    if confidence > 85:
        send_email()
        print('email sended')
        send_whatsapp()
        print('whats message sent')
        create_ec2()
        print('created ec2 create and attached')
        break

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

 

cap.release()
cv2.destroyAllWindows()

