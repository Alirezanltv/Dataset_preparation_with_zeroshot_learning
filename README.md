# Dataset Preparation with Zero-Shot Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLO-World](https://img.shields.io/badge/YOLO--World-Ultralytics-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

This repository provides tools for creating custom object detection datasets using YOLO-World's zero-shot detection capabilities. It allows you to:

1. Search and download images for any topic via DuckDuckGo
2. Detect objects using natural language prompts (no training required)
3. Generate annotated images with bounding boxes
4. Create organized datasets with detection metadata

The project includes both a command-line interface and a user-friendly Streamlit web application.

## ✨ Features

- **Zero-shot object detection** - no training needed for new object classes
- **Automated image collection** from DuckDuckGo search
- **Adaptive confidence thresholds** to optimize detection results
- **Custom NMS implementation** for filtering overlapping detections
- **Detailed visualizations** with color-coded bounding boxes
- **Comprehensive detection statistics** in JSON format
- **Interactive Streamlit UI** for easy use without coding

## 🛠️ Installation

1. Clone this repository:
```bash
git clone https://github.com/Alirezanltv/Dataset_preparation_with_zeroshot_learning.git
cd Dataset_preparation_with_zeroshot_learning
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 📋 Requirements

- Python 3.8+
- ultralytics
- streamlit
- duckduckgo-search
- opencv-python
- Pillow
- numpy
- requests

Create a `requirements.txt` file with:
```
ultralytics>=8.0.0
streamlit>=1.24.0
duckduckgo-search>=3.0.0
opencv-python>=4.5.0
Pillow>=9.0.0
numpy>=1.20.0
requests>=2.25.0
```

## 🚀 Usage

### Command-Line Interface

Run the script to download images and detect objects:



The script will:
1. Ask you which topic you want to search for images
2. Download images based on your topic
3. Ask you to enter detection prompts (object classes)
4. Apply YOLO-World detection on the downloaded images
5. Save detection results with bounding boxes

### Streamlit UI

For a more user-friendly interface, run the Streamlit app:

```bash
streamlit run UI_dataset_preparation.py
```

This will open a web interface where you can:
1. Initialize the YOLO-World model
2. Enter a search topic and download images
3. Specify objects to detect with natural language
4. View detection results with side-by-side comparisons
5. Access saved results and visualizations

## 📊 Project Structure

When you run the application, it creates the following directory structure:

```
YOLO_World_Detection/
└── your_topic_name/
    ├── [downloaded images]
    ├── detections/
    │   └── [json files with detection data]
    ├── visualizations/
    │   └── [annotated images with bounding boxes]
    └── your_topic_name_yolo_summary.json
```

## 🔍 How It Works

1. **Image Collection**: Downloads images from DuckDuckGo based on your search query
2. **Object Detection**: Uses YOLO-World's zero-shot capabilities to detect objects based on text prompts
3. **Detection Optimization**: Tests multiple confidence thresholds to find optimal detection results
4. **Visualization**: Creates annotated images with bounding boxes around detected objects
5. **Data Organization**: Saves detection metadata in JSON format for further use

## 💡 Example

```python
# Command line example - the program will interactively ask for:
# 1. Search topic: "construction site"
# 2. Detection classes: "worker, machinery, scaffold"

# The program will download images of construction sites and detect workers, machinery, 
# and scaffolds, saving the results with bounding boxes.
```

## ⚠️ Limitations

- Dependent on DuckDuckGo search API availability
- Zero-shot detection may not be as accurate as models trained specifically for certain objects
- Image quality and context affect detection performance
- Limited to detecting objects that YOLO-World can recognize from text descriptions

## 🔮 Future Work

- Add support for additional image sources
- Implement dataset export in COCO/YOLO format
- Add active learning capabilities to improve detection quality
- Support for custom fine-tuning of models

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- This project uses [YOLO-World](https://github.com/ultralytics/ultralytics) from Ultralytics
- Search functionality provided by [DuckDuckGo Search](https://github.com/deedy5/duckduckgo_search)

---

Made with ❤️ by [Alireza Kanani and Ramin eslami](https://github.com/Alirezanltv)
