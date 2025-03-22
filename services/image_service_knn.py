from fastapi import HTTPException 
import cv2
import numpy as np
from ..image_processing import detect_and_label_digits

async def process_image(file):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    labeled_image = detect_and_label_digits(image)
    return labeled_image