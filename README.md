# Advanced Face & Feature Detection App

This advanced facial detection and recognition application is built with Streamlit and OpenCV, with additional capabilities provided by DeepFace and deep learning models.

## Features

- **Face Detection**: Uses OpenCV DNN with a pre-trained SSD MobileNet model for accurate face detection with adjustable confidence thresholds.
- **Face Recognition**: Register faces and identify them in new images or real-time video. The system stores facial embeddings for multiple models.
- **Facial Analysis**: Detects facial attributes including age, gender, and emotion using DeepFace's pre-trained models.
- **Feature Detection**: Identifies additional facial features including eyes and smiles using Haar Cascade Classifiers with adjustable parameters.
- **Comparison Mode**: Compare faces between two images using two methods:
  - HOG (Histograms of Oriented Gradients) - fast and effective for quick comparisons
  - Embeddings (deep neural networks) - slower but more precise for accurate matching
- **Video Processing**: Process uploaded videos to detect faces and facial features with frame-by-frame analysis.
- **Interactive UI**: Multiple UI components including sliders, tabs, expandable sections, and dynamic metric displays.
- **Performance Metrics**: View processing time and detection statistics for optimization and insight.
- **Downloadable Results**: Export processed images and videos with annotations for further use.
- **Registered Faces Management**: Descriptive table with the ability to delete individual records or all faces at once.

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/face-detection-app.git
cd face-detection-app
```

2. Install required dependencies:
```
pip install -r requirements.txt
```

3. Required model files:
   The application uses several pre-trained models:
   
   a) **Face Detection Model**: The application looks for these files in the project directory:
      - `deploy.prototxt`
      - `res10_300x300_ssd_iter_140000_fp16.caffemodel`
   
   These files are automatically downloaded when running the application for the first time. However, you can manually download them:
   ```
   # Create a models directory (optional)
   mkdir -p models
   
   # Download the prototxt file
   curl -o models/deploy.prototxt https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
   
   # Download the caffemodel file
   curl -o models/res10_300x300_ssd_iter_140000_fp16.caffemodel https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000_fp16.caffemodel
   ```

   b) **Haar Cascade Classifiers**: These are included with OpenCV and loaded automatically.
   
   c) **DeepFace Models**: These are downloaded automatically when first using facial recognition features.

4. If you encounter any issues with automatic downloads, the models can be found in these repositories:
   - OpenCV DNN Face Models: https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
   - OpenCV Haar Cascades: https://github.com/opencv/opencv/tree/master/data/haarcascades

## Requirements

The application requires the following main dependencies:
```
streamlit>=1.31.0
opencv-python-headless>=4.8.0
numpy>=1.26.0
Pillow>=10.0.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
pandas>=1.3.0
deepface>=0.0.79
tensorflow>=2.8.0
scipy>=1.7.0
mtcnn>=0.1.0
retina-face>=0.0.1
requests>=2.25.0
dlib>=19.22.0
```

## Usage

Run the Streamlit app:
```
streamlit run streamlit_app.py
```

The application will open in your default web browser.

### Application Modes

1. **About**: Overview of the application and its features, with detailed explanations of each module.
2. **Face Detection**: Upload images or videos to detect faces using the OpenCV DNN module.
3. **Feature Detection**: Detect faces along with additional facial features (eyes, smiles) using Haar Cascade Classifiers.
4. **Comparison Mode**: Compare faces between two images using either HOG or Embeddings methods with detailed similarity metrics.
5. **Face Recognition**: Register faces and identify them in new images or real-time video with confidence scores.

### Controls

- Use the sidebar to navigate between different modes and adjust global settings
- Adjust confidence thresholds for detection with sliders (from 0.0 to 1.0)
- Enable/disable specific facial feature detection (eyes, smiles)
- Configure various display options including bounding box colors and line thickness
- Manage registered faces with the descriptive table, with options to delete individual entries or all records

## Implementation Details

### Technologies Used

- **Streamlit**: For the web interface and interactive components, providing a responsive UI
- **OpenCV**: For computer vision operations including face and feature detection using DNN and Haar Cascades
- **NumPy**: For efficient array operations and numerical computations
- **PIL (Pillow)**: For image processing, manipulation, and format conversion
- **DeepFace**: For advanced facial recognition and attribute analysis using deep learning models
- **TensorFlow**: For deep neural network models that power the facial recognition system
- **scikit-learn**: For machine learning algorithms and similarity comparison using cosine distance
- **pandas**: For tabular data handling and management of facial databases

### Model Information

- **Face Detection**: OpenCV DNN module with SSD MobileNet, trained on the WIDER FACE dataset
- **Face Recognition**: Pre-trained DeepFace models:
  - VGG-Face: Based on the VGG-16 architecture, trained on a large-scale facial recognition dataset
  - Facenet: Google's FaceNet model, using a deep convolutional network
  - OpenFace: An open-source facial recognition model based on FaceNet architecture
  - ArcFace: State-of-the-art facial recognition model using Additive Angular Margin Loss
- **Eye Detection**: Haar Cascade Classifier (haarcascade_eye.xml), trained on positive and negative eye images
- **Smile Detection**: Haar Cascade Classifier (haarcascade_smile.xml), for detecting smiles with adjustable parameters

### User Interface Enhancements

The application includes several UI improvements:
- Tab-based navigation for different functionalities, allowing easy switching between modes
- Sidebar with control settings for global parameters that affect all modes
- Progress bars for video processing to show completion status
- Metric displays for performance statistics, including processing time and detection counts
- Expandable sections for detailed information, reducing visual clutter
- Color pickers for customizing display options like bounding box colors
- Descriptive table for managing registered faces with clear information on embeddings and models

## Advanced Features

### Improved Similarity Algorithm
The application includes an enhanced similarity algorithm for facial comparison that:
- Uses a stronger power curve (1.3) for better discrimination between similar and dissimilar faces
- Gives more weight to precise facial structure (25%) for improved matching accuracy
- Applies more aggressive reductions for low similarities to better differentiate non-matches
- Introduces a "critical difference score" that can reduce similarity by up to 25% when significant differences are detected

### Updated Similarity Thresholds
New, stricter similarity ranges for more accurate matching:
- HIGH (80-100%): Very likely the same person
- MEDIUM (65-80%): Possibly the same person
- LOW (35-65%): Unlikely to be the same person
- VERY LOW (0-35%): Different people

### Face Database Management
- Registration of multiple embeddings per person for improved recognition across different conditions
- Support for different embedding models (VGG-Face, Facenet, OpenFace, ArcFace)
- Interactive face management interface with tabular display
- Functionality to delete individual records or all faces at once for database maintenance

## Future Enhancements

- Improve facial recognition accuracy with ensemble methods combining multiple models
- Add face tracking in videos for consistent identification across frames
- Implement user authentication for secure access to the face database
- Add capability to export/import the face database for backup and transfer
- Enhance result visualization with advanced graphs and interactive reports
- Implement batch processing for large datasets of images or videos

## Credits

- Based on the tutorial by Dr. Ernesto Lee
- Uses pre-trained models from OpenCV and DeepFace
- Built with Streamlit and Python

