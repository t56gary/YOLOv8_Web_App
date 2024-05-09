# YOLOv8 Web App

## Project Overview
This project integrates a FastAPI backend with a Streamlit frontend to create a web-based application for uploading images, processing them with the YOLOv8 model for object detection, and displaying the results. The application supports image uploads in Base64 encoding, processes them using a pretrained YOLOv8 model, and returns processed image paths for display.

## Features
- **Image Upload**: Users can upload images via a Streamlit interface.
- **Object Detection**: Uses the YOLOv8 model to detect objects in uploaded images.
- **Result Visualization**: Processed images with detected objects are displayed.

## Technologies Used

- **FastAPI**: Backend server.
- **Streamlit**: Frontend interface.
- **YOLOv8 (Ultralytics)**: Object detection.
- **Pydantic**: Data validation.
- **Python**: Programming language.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourrepository/yourproject.git
   cd yourproject
2. **Install Required Libraries**
   ```bsah
   pip install fastapi uvicorn streamlit ultralytics[aio] pydantic aiofiles
3. **Environment Setup**
   - Ensure Python 3.8+ is installed.
   - CUDA environment (if running YOLOv8 with GPU support).

## Usage

1. **Start the Backend Server**
   ```bash
   uvicorn backend:app --reload  # Replace 'backend' with your FastAPI script name if different
   
2. **Run the Streamlit Frontend**
   ```bash
   streamlit run frontend.py  # Replace 'frontend.py' with your Streamlit script name

3. **Interacting with the Application**
   - Navigate to the Streamlit URL (usually http://localhost:8501).
   - Use the file uploader to select and upload an image.
   - View the processed image and detection results displayed on the page.
   - The output of the YOLOv8 detection will be saved as a JSON file in the same folder where the backend is running.