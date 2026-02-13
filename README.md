# Implementation of SIFT for ID Card Object Recognition
This repository contains the source code for the Final Project of the Computer Vision course. The project implements the **Scale-Invariant Feature Transform (SIFT)** algorithm to recognize and localize an Identity Card (ID Card) using Python and OpenCV.

## 📌 Project Overview
The system is designed to detect a specific object (ID Card) from a reference image in a target scene. It is robust against:
* **Scale variations** (distance changes)
* **Rotational changes** (different angles)
* **Partial occlusion** (part of the card being covered)

## 🛠️ Installation & Requirements (Reproducibility)
To replicate this project on your local machine, follow these steps:

1. **Prerequisites**: Ensure you have Python 3.x installed.
2. **Install Libraries**: Run the following command to install the required dependencies:
   ```bash
   pip install opencv-python numpy matplotlib
