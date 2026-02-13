# Implementation of SIFT for ID Card Object Recognition
This repository contains the source code for the Final Project of the Computer Vision course. The project implements the **Scale-Invariant Feature Transform (SIFT)** algorithm to recognize and localize an Identity Card (ID Card) using Python and OpenCV.

## 📌 Project Overview
The system is designed to detect a specific object (ID Card) from a reference image in a target scene. Based on the experimental results, the system achieved:
* **Keypoints detected**: ~2,800 points (Reference) and ~3,000 points (Target).
* **Robust Matching**: 490 good matches after Lowe's Ratio Test.
* **Accuracy**: Successful Homography with 337/490 inliers.

## 🛠️ Installation & Requirements (Reproducibility)
To replicate this project on your local machine, follow ini:

1. **Prerequisites**: Ensure you have Python 3.x installed.
2. **Install Libraries**: Run the following command to install the required dependencies:
   ```bash
   pip install opencv-contrib-python numpy matplotlib
