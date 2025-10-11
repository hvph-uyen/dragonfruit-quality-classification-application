# DRAGON FRUIT QUALITY CLASSIFICATION APPLICATION
## 1. Introduction

This project applies Deep Learning models to detect and classify dragon fruits via images. The application is divided into two main parts:
- Backend (Python): Handles image processing and runs the machine learning models.
- Frontend (React + TypeScript): Allows users to upload images and view prediction results.

The project uses two pre-trained .pth models:
- Detection model: Locates dragonfruits within an image.
- Classification model: Identifies the type of dragonfruit (`reject`, `good`, `immature`).

## 2. Project structure
```
├── dist/
│ ├── assets/ # static compiled assets
│ └── index.html
│
├── dragon-detection/
│ ├── model/ # Pre-trained models
│ │ ├── classifier_model2.pth # classification model
│ │ └── detector_model.pth # detection model
│ ├── model.py # model source code
│ ├── prepareData.py # data preparation script
│ ├── tsconfig.json # TypeScript configuration
│ └── use_model.py # inference script for model usage
│
├── src/ 
│ ├── App.tsx # main React component
│ ├── ImageUploader.tsx # component for uploading images
│ ├── index.css # global styles
│ └── main.tsx # React entry point
│
├── .gitignore #
├── README.md # this file
├── index.html # HTML template for the app
├── package-lock.json 
├── package.json # Node.js dependencies and scripts
├── requirements.txt # Python dependencies
├── server.py # backend server
└── tsconfig.json # TypeScript configuration for frontend
```

## 3. Installation & Setup

### a. Set up the Python environment

Open a terminal and install the required libraries:

```pip install -r requirements.txt```

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

- Open the web app at `http://localhost:5173`

- Upload an image containing dragon fruits

- The system will process the image, display detected fruits with bounding boxes, and classify each fruit

- Try uploading different images to test performance

## 7. Author Information

- Team name: JOILBEE
- Contact email: thitkhomamruot7749@gmail.com

Created September 2025.
