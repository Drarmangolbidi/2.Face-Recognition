#up
original_image=cv2.imread('picture.jpg')

messi_image = face_recognition.load_image_file('messi.jpeg')
messi_face_encoding = face_recognition.face_encodings(messi_image)[0]

ronaldo_image = face_recognition.load_image_file('ronaldo.jpg')
ronaldo_face_encoding = face_recognition.face_encodings(ronaldo_image)[0]

known_face_encodings = [messi_face_encoding , ronaldo_face_encoding ]
known_face_names = [ 'Lionel Messi' , 'Cristiano Ronaldo']

image_to_recognize = face_recognition.load_image_file('picture.jpg')

all_face_locations = face_recognition.face_locations(image_to_recognize , model='hog')

all_face_encodings = face_recognition.face_encodings(image_to_recognize , all_face_locations)

print('There Are {} Number of Faces in This Image'.format(len(all_face_locations)))


# In[5]:


for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
    
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


# In[ ]:




