import cv2
import pandas as pd
from datetime import date
from simple_facerec import SimpleFacerec

#Function to get todays date
today = date.today()
d1 = today.strftime("%d/%m/%Y")


# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

#opening and reading data from excel sheet
data = pd.read_excel('attendance.xlsx')
l=[]

for i in range(len(data['Name'])):
    l.append(0)
#print(l)
data[str(d1)] = l
# data.insert(len(data['Name']), str(d1), l)


# Load Camera
# url = "http://100.102.138.78:8080/video"
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    
    # Detect Faces
    
    #print(ret, frame)
    if(ret):
        face_locations, face_names = sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
            for i in range(len(data['Name'])):
                if data['Name'][i] == name :
                    data[str(d1)][i] = 1
        frame = cv2.resize(frame, (1280, 1080))
        cv2.imshow("Frame", frame)

    key = cv2.waitKey(30)
    if key == 27 or key == ord('x'): 
        break

cap.release()
cv2.destroyAllWindows()
data.to_excel('attendance.xlsx', index=False)