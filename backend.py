from fastapi import FastAPI
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from pydantic import BaseModel
from data import Rectangle , BasePoint
import aiofiles
import os
import base64
import json
import torch


# declare upload type
class Upload(BaseModel):
    files: str  # Base64 string
    name: str

# declare fastapi
app = FastAPI()

# Load the YOLOv8 model
model = YOLO('yolov8n') 

def process_yolov8_output(boxes, names, filename):
    detections_json = []
    for i in range(boxes.data.shape[0]):
        bbox = boxes.data[i].cpu().numpy()  # Convert from tensor and handle GPU data
        cls_id = int(boxes.cls[i].item())  # Convert tensor to integer
        conf = float(boxes.conf[i].item())  # Convert tensor to float

        x1, y1, x2, y2 = bbox[:4]
        width = x2 - x1
        height = y2 - y1
        center_x = x1 + width / 2
        center_y = y1 + height / 2

        rectangle = Rectangle(
            name=names[cls_id],
            c=BasePoint(x=center_x, y=center_y),
            wh=BasePoint(x=width, y=height)
        )
        rectangle_json = rectangle.model_dump()
        detections_json.append(rectangle_json)

    os.makedirs(os.path.dirname("output/"), exist_ok=True)

    # Write to JSON file
    with open('./output/%s.json'%filename, 'w',) as f:
        json.dump(detections_json, f, indent=4)

# declare upload function
@app.post("/upload")
async def upload(upload: Upload):
    # Decode the file from base64 string into bytes
    file_bytes = base64.b64decode(upload.files)
    # Decode the file path
    file_location = f"temp/{upload.name}"
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_location), exist_ok=True)

    # Save the file asynchronously
    async with aiofiles.open(file_location, 'wb') as out_file:
        await out_file.write(file_bytes)

    # Run YOLO model detection
    results = model.predict(file_location, save=True,device=("cuda:0" if torch.cuda.is_available() else "cpu"))

    # Save the results into json format
    process_yolov8_output(results[0].boxes, results[0].names,os.path.splitext(upload.name)[0])

    # Handling results and extracting saved directory
    save_directories = [result.save_dir for result in results if hasattr(result, 'save_dir')]
    
    # Return the saved directory as a JSON response
    return JSONResponse(content={"img_path": save_directories[0] if save_directories else None})

