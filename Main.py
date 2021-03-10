import cv2
import numpy as np
import face_recognition
import os

path = 'ImageAttendance'
images = []
classNames = []
myList = os.listdir(path)
# print(myList)

# put into the list array
for cls in myList:
  currentImg = cv2.imread(f'{path}/{cls}')
  images.append(currentImg)
  classNames.append(os.path.splitext(cls)[0])
# print(classNames)

# func to generate encodings of each face
def findEncodings(images):
  encodeList = []
  for img in images:
    print('.', end=' ')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodedImg = face_recognition.face_encodings(img)[0]
    encodeList.append(encodedImg)
  return(encodeList)

encodeListKnown = findEncodings(images)
print('Encoding Complete')


webcam = cv2.VideoCapture(0)
webcam.set(3, 840) # ubah lebar cam
webcam.set(4, 680) # ubah tinggi cam

# take each frame of video capture
while True:
  success, img = webcam.read()
  imgSmaller = cv2.resize(img, (0, 0), None, 0.25, 0.25) # reduce size
  imgSmaller = cv2.cvtColor(imgSmaller, cv2.COLOR_BGR2RGB) # convert BGR to RGB

  # might find multiple faces, therefore use face_locations()
  facesCurFrame = face_recognition.face_locations(imgSmaller)
  encodesCurFrame = face_recognition.face_encodings(imgSmaller, facesCurFrame)
  
  # compare faces with database (iterate through all faces that found in webcam)
  for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
    print(faceDis)
    matchIndex = np.argmin(faceDis) # mengambil index dgn value paling kecil

    if int(matches[matchIndex]):
      name = classNames[matchIndex].upper()
      print(name)
      y1, x2, y2, x1 = faceLoc
      y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4 # mengembalikan ke posisi semula (setelah resize)
      cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
      cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 255), cv2.FILLED)
      cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
  

  cv2.imshow('Webcam', img)

  keyTerminate = cv2.waitKey(1) & 0xFF
  if keyTerminate == 27 or keyTerminate == ord('q'):
    break
