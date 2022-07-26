# 2.Face-Recognition
In this project we use Hog and face_recognise to detect my colleagues.

<br>

I upload my Friends photo and I detect them in photo .
<br>

## :blush:Step Two :( Detect Video ):blush:</b>

<br>
Download program "1.Image Face Recognition.py" and these images "messi-neymar.jpg" & "messi-ronaldo.jpg" & "messi.jpeg" & "messi.jpg" & "picture.jpg" & "ronaldo-ramos.jpg" & "ronaldo.jpg" . 
<br>

Code is :👇

<br>

```python
import cv2
import face_recognition
#Libraries 

original_image=cv2.imread('picture.jpg')
#Import image

messi_image = face_recognition.load_image_file('messi.jpeg')
messi_face_encoding = face_recognition.face_encodings(messi_image)[0]
#load image of user in face_recognition
#Convert Image to blob

ronaldo_image = face_recognition.load_image_file('ronaldo.jpg')
ronaldo_face_encoding = face_recognition.face_encodings(ronaldo_image)[0]
#load image of user in face_recognition
#Convert Image to blob

known_face_encodings = [messi_face_encoding , ronaldo_face_encoding ]
known_face_names = [ 'Lionel Messi' , 'Cristiano Ronaldo']
#Save User blobs in dictionary and after that save name of user in dictionary . 

image_to_recognize = face_recognition.load_image_file('picture.jpg')
#Upload the photo to facerecognition 

all_face_locations = face_recognition.face_locations(image_to_recognize , model='hog')
#Find location of face in image with hog 

all_face_encodings = face_recognition.face_encodings(image_to_recognize , all_face_locations)
#convert photo to blob

print('There Are {} Number of Faces in This Image'.format(len(all_face_locations)))
#find number of face in image

for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
#zip function return iterator like this Shape ((ali,arman),(reza,Gholi)) 
    top ,right ,bottom ,left  = current_face_location
    
    all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
   
    name_of_person = 'Unknown face'
    
    if True in all_matches:
        first_match_index = all_matches.index(True)
        name_of_person = known_face_names[first_match_index]
    
    cv2.rectangle(original_image,(left,top),(right,bottom),(255,0,0),2)
    
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(original_image, name_of_person, (left,bottom), font, 0.5, (255,255,255),1)
    
    cv2.imshow("Faces Identified",original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



```
#### EX11_Leve :
- [ ] Simple! 
- [ ] Intermediate!
- [x] Hard!

<br>
