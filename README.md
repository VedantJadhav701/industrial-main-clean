# 🔍 Steel Defect Segmentation - Industrial Quality Control

An advanced deep learning project for automated steel surface defect detection and segmentation using PyTorch and Streamlit.

## 🚀 Features

- **Advanced U-Net Architecture** with attention mechanisms
- **Real-time Web Interface** built with Streamlit
- **High Accuracy Detection** (98.79% pixel accuracy)
- **4-Class Defect Segmentation** for different steel defect types
- **Interactive Visualization** with confidence scores
- **CUDA Support** for GPU acceleration

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Pixel Accuracy | 98.79% |
| Dice Score | 88.38% |
| IoU Score | 87.16% |
| F1 Score | 92.43% |
| Precision | 94.68% |
| Recall | 90.76% |

### Per-Class IoU Performance
- **Defect 1**: 99.24%
- **Defect 2**: 99.96%
- **Defect 3**: 88.74%
- **Defect 4**: 94.98%

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch, TorchVision
- **Computer Vision**: OpenCV, PIL, Albumentations
- **Web Framework**: Streamlit
- **Data Science**: NumPy, Pandas, Matplotlib, Seaborn
- **ML Utils**: Scikit-learn, Segmentation Models PyTorch

## 📁 Project Structure

```
Steel-Segmentation-Defect-Detect/
├── app.py                          # Streamlit web application
├── main.ipynb                      # Main training notebook
├── h.ipynb                         # Model development notebook
├── benchmark.ipynb                 # Performance benchmarking
├── data/
│   ├── train_images/               # Training images
│   ├── test_images/                # Test images
│   ├── train.csv                   # Training labels
│   └── sample_submission.csv       # Submission format
├── outputs/
│   ├── checkpoints/                # Model checkpoints
│   │   └── best_model_dice.pth    # Best trained model
│   ├── metrics/                    # Training metrics
│   ├── model/                      # Model artifacts
│   └── visualizations/             # Generated plots
└── projects/
    └── Steel-Segmentation-Defect-Detect/  # Additional resources
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (recommended)
- Conda or Miniconda

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/VedantJadhav701/Steel-Segmentation-Defect-Detect.git
   cd Steel-Segmentation-Defect-Detect
   ```

2. **Create conda environment**
   ```bash
   conda create -n ml_env python=3.11
   conda activate ml_env
   ```

3. **Install dependencies**
   ```bash
   # PyTorch with CUDA support
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   
   # Other dependencies
   conda install streamlit numpy pillow matplotlib opencv scikit-learn pandas seaborn
   pip install albumentations segmentation-models-pytorch
   ```

4. **Download pre-trained models**
   
   Due to GitHub's file size limitations (~360MB), model weights need to be downloaded separately:
   
   **Option A: Train your own model**
   ```bash
   # Run the training notebook
   jupyter notebook main.ipynb
   # Follow the training cells to generate model checkpoints
   ```
   
   **Option B: Use provided models** (if available)
   - Download `best_model_dice.pth` from releases or shared drive
   - Place in `outputs/checkpoints/` directory
   
   **Check setup**
   ```bash
   python check_setup.py  # Verify all components are ready
   ```
   
   **Option B: Using download script**
   ```bash
   python download_models.py
   ```
   
   **Note**: Update the URLs in `download_models.py` with your hosted model files.

### Running the Application

1. **Start Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Upload a steel image** and get instant defect segmentation results!

## 🎯 Model Architecture

- **Base**: Improved U-Net with Attention Mechanisms
- **Encoder**: Convolutional blocks with BatchNorm and Dropout
- **Decoder**: Transposed convolutions with skip connections
- **Attention**: Gate attention for better feature focus
- **Output**: 4-channel segmentation (one per defect type)

## 📈 Training Details

- **Dataset**: Steel surface defect images with RLE-encoded masks
- **Augmentation**: Albumentations (rotation, flip, brightness, etc.)
- **Loss**: Combined BCE + Dice + Focal Loss
- **Optimizer**: AdamW with Cosine Annealing
- **Training**: ~31M parameters, CUDA acceleration

## 🔧 Usage

### Web Interface
1. Upload steel surface image (PNG/JPG)
2. View segmentation masks for 4 defect types
3. See confidence scores and detection metrics
4. Analyze defect coverage percentage

### Programmatic Usage
```python
import torch
from app import ImprovedUNet

# Load trained model
model = ImprovedUNet(in_ch=3, out_ch=4)
checkpoint = torch.load('outputs/checkpoints/best_model_dice.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Run inference
with torch.no_grad():
    prediction = model(image_tensor)
    masks = torch.sigmoid(prediction)
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Steel defect dataset providers
- PyTorch and Streamlit communities
- Open source computer vision libraries

## 📞 Contact

**Vedant Jadhav** - [@VedantJadhav701](https://github.com/VedantJadhav701)

Project Link: [https://github.com/VedantJadhav701/Steel-Segmentation-Defect-Detect](https://github.com/VedantJadhav701/Steel-Segmentation-Defect-Detect)

---

⭐ **Star this repository** if you find it helpful!