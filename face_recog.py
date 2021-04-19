import face_recognition 
import numpy as np

def load_known_face(target_path):
    # Load the sample picture and learn how to recognize it
    print("Loading known faces")
    target_image=face_recognition.load_image_file(target_path)
    target_face_encoding=face_recognition.face_encodings(target_image)[0]
    return target_face_encoding

def face_recog(frame,target_face_encoding):
    
    # Initialize some variables
    face_loactions=[]
    face_encodings=[]
    faces=[]
    print("Capturing image!")
    #Find all the faces and face encodings in the current frame of Video
    face_locations=face_recognition.face_locations(frame)
    print("Found {} faces in image.".format(len(face_locations)))
    face_encodings=face_recognition.face_encodings(frame,face_locations)
        
    #Loop over each face found in the frame to see if it's someone we know.
        
    for face_encoding, face_location in zip(face_encodings,face_locations):
        #see if the face is a match for the known faces
        match=face_recognition.compare_faces([target_face_encoding],face_encoding)
        name="<Unknown Person>"
            
        if match[0]:
            name="Target"
            faces.append(face_location)
            print(face_location)
        print("I see someone named {}!".format(name))
        
        
        
    return faces
            