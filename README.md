 Fruit Image Classification Using Convolutional Neural Network (CNN)

A deep learning–based fruit image classification system using a custom Convolutional Neural Network (CNN). This project is designed for smart agriculture, automated fruit sorting, and food quality inspection, focusing on both high accuracy and computational efficiency.

Project Highlights
•	Custom CNN architecture trained from scratch
•	Automatic feature extraction (no handcrafted features)
•	High classification accuracy (≈96%+)
•	Lightweight and deployable model
•	Suitable for academic and real-world use

Project Structure
fruit_classification/
├── data/
│   ├── train/            # Training images
│   └── validation/       # Validation images
├── models/
│   └── fruit_classifier.h5
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_training.ipynb
├── src/
│   ├── data_loader.py
│   ├── model.py
│   └── utils.py
├── config.py
├── requirements.txt
└── README.md

 CNN Model Architecture
•	Input: 128×128×3 RGB image
•	Convolution Blocks: 4 blocks (32 → 64 → 128 → 256 filters)
•	Activation: ReLU
•	Regularization: Batch Normalization + Dropout
•	Classifier: Fully Connected Dense Layers
•	Output: Softmax (multi-class fruit classification)
This architecture balances accuracy, speed, and model size, making it suitable for deployment on resource-constrained systems.

 System Requirements
Operating System
•	Windows 10 / 11 (64-bit)
•	Ubuntu 18.04 or later
•	macOS Catalina or later
Hardware
Minimum:
•	CPU: Intel i5 or equivalent
•	RAM: 8 GB
•	Storage: 10 GB free
Recommended:
•	CPU: Intel i7 / AMD Ryzen 7
•	RAM: 16 GB+
•	GPU: NVIDIA GPU (CUDA-enabled, 4 GB+ VRAM)

🛠️ Software & Tools
•	Python 3.8+
•	TensorFlow / Keras
•	NumPy
•	Pandas
•	Matplotlib
•	Seaborn
•	OpenCV
•	Scikit-learn
•	Jupyter Notebook
Install dependencies:
pip install -r requirements.txt

Dataset Details
•	Image format: JPG / PNG
•	Input size: Resized to 128×128×3
•	Dataset split:
o	Training set
o	Validation set
•	Folder structure: Class-wise directories

How to Run
1.	Clone the repository
git clone <repository-url>
cd fruit_classification
2.	Install dependencies
pip install -r requirements.txt
3.	Add dataset to data/ directory
4.	Open and run training notebook
jupyter notebook notebooks/model_training.ipynb
5.	Evaluate model performance

Output
•	Trained model: fruit_classifier.h5
•	Training & validation accuracy plots
•	Performance metrics (accuracy, loss)

🔮 Future Enhancements
•	Transfer learning integration (MobileNet, ResNet)
•	Real-world dataset expansion
•	Mobile / edge device deployment
•	Fruit quality and ripeness detection
 License
This project is intended for academic and educational purposes only.

 If you find this project useful, consider starring the repository!
<img width="468" height="643" alt="image" src="https://github.com/user-attachments/assets/bacf5598-5873-4432-a64a-e6f850d03769" />
