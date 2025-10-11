# DRAGON FRUIT QUALITY CLASSIFICATION APPLICATION
## 1. Introduction

This project applies Deep Learning models to detect and classify dragon fruits via images. The application is divided into two main parts:
- Backend (Python): Handles image processing and runs the machine learning models.
- Frontend (React + TypeScript): Allows users to upload images and view prediction results.

The project uses two pre-trained .pth models:
- Detection model: Locates dragonfruits within an image.
- Classification model: Identifies the type of dragonfruit (`reject`, `good`, `immature`).

## 2. Folder Structure
```
├── dist/
│ ├── assets/
│ └── index.html
│
├── dragon-detection/
│ ├── model/
│ │ ├── classifier_model2.pth
│ │ └── detector_model.pth
│ ├── model.py
│ ├── prepareData.py
│ ├── tsconfig.json
│ └── use_model.py
│
├── src/
│ ├── App.tsx
│ ├── ImageUploader.tsx
│ ├── index.css
│ └── main.tsx
│
├── .gitignore
├── README.md
├── index.html
├── package-lock.json
├── package.json
├── requirements.txt
├── server.py
└── tsconfig.json
```

## 3. Installation & Setup

### a. Set up the Python environment

Open a terminal and install the required Python libraries:

```pip install <library_names>```

### b. Run the backend server

After installing the dependencies, start the backend with:

```python server.py```

### c. Install and run the frontend (React)

Navigate to the frontend directory (where package.json is located), then run:
```
npm install
npm install react react-dom
npm start
```
The web application will be available at: http://localhost:5173

## 4. Model Information

`classifier_model2.pth`: Classifies dragon fruits into categories. The model was trained on real-world image datasets.

`detector_model.pth`: Detects and draws bounding boxes around dragon fruits in images.

## 5. System Requirements

- Python: Version 3.8 or higher
- Node.js: Version 14 or higher

## 6. Usage Instructions

a. Open the web app at `http://localhost:5173`

b. Upload an image containing dragon fruits

c. The system will process the image, display detected fruits with bounding boxes, and classify each fruit

d. Try uploading different images to test performance

## 7. Required Python Libraries

- torch
- numpy
- opencv-python
- pillow

## 8. Author Information

- Team Name: JOILBEE
- Contact Email: thitkhomamruot7749@gmail.com

Created September 2025
