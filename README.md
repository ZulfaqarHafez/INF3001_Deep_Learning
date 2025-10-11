# INF3001_Project  

## Beyond Helmets: Attention-Augmented, Domain-Robust Multi-Label PPE Recognition

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)  [![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)](https://fastapi.tiangolo.com/)  [![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange?logo=pytorch)](https://pytorch.org/)  [![Frontend](https://img.shields.io/badge/Frontend-HTML%2FCSS%2FJS-yellow?logo=javascript)]()  [![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Project Overview

This project focuses on the **automatic detection of Personal Protective Equipment (PPE)** usage in hazardous work environments.  
Manual inspection is **time-consuming, error-prone, and difficult to scale** across large or dynamic environments.  

By leveraging **deep learning and computer vision**, this project aims to automate PPE compliance monitoring through image and video analysis providing a scalable, intelligent solution for workplace safety.

---

## Objectives
- Develop an AI-based image classification model to detect **helmet usage** in hazardous environments.  
- Incorporate **attention mechanisms (CBAM)** to enhance feature extraction under occlusion or clutter.  
- Apply **transfer learning** for efficient training and improved generalization.  
- Build a **web application** for real-time PPE compliance feedback.  
- Extendable to multi-label PPE (helmets, vests, gloves, masks) for future scalability.

---

## Methodology

### **1️: Dataset Preparation**
We procured and organized a helmet/no-helmet dataset, split into **train, validation, and test sets** using stratified sampling.  
Strong augmentations (rotation, color jitter, random erasing) simulate real-world conditions to boost model robustness.

### **2️:  Model Architecture**
- Baseline: **ResNet-18** with pretrained ImageNet weights  
- Final layer replaced with a **multi-label classification head**  
- Integrated **Convolutional Block Attention Module (CBAM)** to refine spatial and channel features for better PPE localization.

### **3️: Training Strategy**
- **Transfer learning** and fine-tuning  
- **Dropout (0.5)** and **weight decay (1e-3)** for regularization  
- **SGD with momentum (0.9–0.95)** for stable convergence  
- **Mixed precision (AMP)** for performance on GPU

### **4️: Evaluation Metrics**
- Accuracy  
- F1-score  
- Loss curves  
- Grad-CAM visualizations to highlight model attention regions  

---

## Results and Analysis
- Input: **224×224 images**, batch size **32**, trained for **8–15 epochs**  
- Initial Adam runs overfitted (train loss ≈ 0.2, val/test ≈ 0.5, acc ≈ 0.84–0.86)  
- Switching to **SGD + momentum** stabilized training (acc ≈ 0.68–0.92)  
- Best model:  
  - **Test loss:** 0.3118  
  - **Test accuracy:** 0.9222  
- Removing dropout later reduced test accuracy to 0.8611 → confirming **dropout’s regularization benefit**.  

Key takeaway: **Dataset quality, momentum tuning, and dropout** had the largest impact on performance.

---

## Conclusions and Reflection
This project demonstrates the feasibility of **deep learning–based PPE compliance monitoring** using **attention-enhanced ResNet-18**.  
By improving robustness under occlusion and poor lighting, the framework serves as a foundation for broader PPE detection (masks, vests, gloves).  

In future iterations:
- Expand dataset to more PPE classes  
- Explore **lightweight architectures** for **edge deployment**  
- Investigate **video-based temporal models** for real-time stability  

---

## Dataset References
1. [Ultralytics Construction PPE Dataset](https://docs.ultralytics.com/datasets/detect/construction-ppe/#business-value)  
2. [Snehil Sanyal – Construction Site Safety PPE Detection (GitHub)](https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection)  
3. [Safety Helmet Wearing Dataset (Roboflow)](https://universe.roboflow.com/zayed-uddin-chowdhury-ghymx/safety-helmet-wearing-dataset/browse)

---

## Classifier Web App

A **FastAPI + PyTorch** powered backend serves real-time PPE classification through an interactive frontend built with HTML/CSS/JS.

### Features
- Drag-and-drop or upload images  
- Real-time inference via **FastAPI**  
- Dynamic prediction card with color-coded feedback (🟢 Helmet / 🔴 No Helmet)  
- Responsive layout for desktop and mobile  
- Smooth animations and progress feedback  

---

## Setup & Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/helmet-classifier.git
cd helmet-classifier
Backend (FastAPI + PyTorch)
```
``` bash
Copy code
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt
cd backend
uvicorn main:app --reload --port 8000
Backend runs at → http://127.0.0.1:8000
API Docs → http://127.0.0.1:8000/docs
```
Frontend
```bash
Copy code
cd frontend
python -m http.server 5500
Open in browser → http://127.0.0.1:5500/frontend/index.html
```
👥 Team
SIT – AAI3001 (Deep Learning and Computer Vision, Tri 1, 2025)

Muhammad Zulfaqār Bin Abdul Hafez (2401578)

Tan Chun Yuan (2400562)

Teo Royston (2402079)

Lim Si Wei Shawn (2400757)

Chua Shan Yang Daniel (2400855)

🧾 License
This project is released under the MIT License.
