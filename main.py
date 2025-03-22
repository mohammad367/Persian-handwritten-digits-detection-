from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse 
import cv2
import numpy as np
import io

from services.image_processor import image_processor

# from image_processing import detect_and_label_digits
# from services.image_service_knn import process_image as knn_image_processor 

app = FastAPI()


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    image = await image_processor(file)
    labeled_image = image# detect_and_label_digits(image)

    _, encoded_image = cv2.imencode(".png", labeled_image)
    return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)