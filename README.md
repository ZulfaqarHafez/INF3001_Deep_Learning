# INF3001_Project
Deep learning project
🦺 PPE Classification Project
📌 Overview

This project is a Personal Protective Equipment (PPE) classification system that can distinguish whether a person is wearing a helmet or not.
It includes:

A PyTorch classifier trained on custom dataset (helmet, no_helmet).

A simple frontend (HTML, CSS, JS) for user interaction.

Jupyter notebook for training the classifier.

⚠️ Note: Database integration (PostgreSQL + MongoDB) will be added later to log detections and support advanced queries.

📂 Project Structure
project-root/
│
├── dataset/                 # Images for training & validation
│   ├── train/
│   │   ├── helmet/
│   │   └── no_helmet/
│   └── val/
│       ├── helmet/
│       └── no_helmet/
│
├── frontend/                # Simple web UI
│   ├── js/
│   ├── public/
│   ├── styles/
│   └── index.html
│
├── notebooks/
│   └── train_classifier.ipynb   # Jupyter notebook for training
│
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md                # Project documentation