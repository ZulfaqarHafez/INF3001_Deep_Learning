# INF3001_Project
Deep learning project
PPE Classification Project
Overview

# Classifier Web App

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)  [![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)](https://fastapi.tiangolo.com/)  [![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange?logo=pytorch)](https://pytorch.org/)  [![Frontend](https://img.shields.io/badge/Frontend-HTML%2FCSS%2FJS-yellow?logo=javascript)]()  [![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Overview
Classifier is a **deep learning–powered web application** that detects whether a person is wearing a helmet from an uploaded image.  

The app integrates:  
- **PyTorch** for model training and inference (`.pth` weights).  
- **FastAPI** backend to serve predictions.  
- **Modern frontend (HTML, CSS, JS)** for an interactive user experience.  

---

## Features
- Upload or drag-and-drop images for classification  
- Real-time inference powered by FastAPI & PyTorch  
- Clear top prediction card with **green (helmet)** or **red (no helmet)** highlight  
- Responsive UI (side-by-side layout on desktop, stacked layout on mobile)  
- Smooth animations and progress feedback  

---

## Setup & Installation

### Clone the repository
```bash
git clone https://github.com/yourusername/helmet-classifier.git
cd helmet-classifier
2️Backend (FastAPI + PyTorch)
Create and activate a virtual environment:
```
```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
Install dependencies:
```
```bash
pip install -r requirements.txt
Run the backend:
```
```bash
cd backend
uvicorn main:app --reload --port 8000
Backend will run at: http://127.0.0.1:8000

Health check → http://127.0.0.1:8000/health
```
API docs → http://127.0.0.1:8000/docs

Frontend
Serve the frontend (any static server works, e.g., Python HTTP or VS Code Live Server):

```bash
cd frontend
python -m http.server 5500
Open in browser:
http://127.0.0.1:5500/frontend/index.html
```
