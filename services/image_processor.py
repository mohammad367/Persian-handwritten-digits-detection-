import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException

from detectors.digit_detector_and_lable_adding import detector

async def image_processor(file):
    label_with_position={}
    contents = await file.read()
    image = np.array(cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR))
    original_image=np.copy(image)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    image=255-image
    print(image.shape)

    num_labesl, labels, stats, centers=cv2.connectedComponentsWithStats(image)

    for label in range(1,num_labesl):
        x,y,w,h= stats[label][0:4]
        if w<10 or h<10:
            continue
        cropped_image=image[y:y+h,x:x+w]
        cropped_image=cv2.resize(cropped_image,(5,5))
        
        detected_label= (detector(cropped_image.reshape(-1,25)))
        label_with_position[(x, y, w, h)]=detected_label
    
   
    for x, y, w, h in label_with_position.keys():
        cv2.rectangle(original_image,(x, y), (x+w, y+h), (255,0,0),2)

        cv2.putText(
            original_image,  
            str(label_with_position[(x,y,w,h)]), 
            (int(x + w / 3), int(y)), 
            cv2.FONT_HERSHEY_SIMPLEX,  
            1, 
            (255, 0, 0),  
            2  
        )
    return original_image