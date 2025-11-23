---
title: Gender Age Race Prediction
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

## End to end Gender Age Race Prediction project.
This repository contains a complete end-to-end machine learning workflow focused on predicting Gender, Age and Race.  
It includes Exploratory Data Analysis (EDA), experimental model training notebooks, a fully modular training pipeline, and a deployable application interface.

### 🔍 What’s Inside
- **EDA notebooks** for exploring dataset structure and understanding feature relationships.
- **Model training notebooks** for experimentation and benchmarking.
- **End-to-end training pipeline** implemented as Python scripts.
- **Gradio web application** for real-time inference.
- **CI/CD pipeline** implemented using **GitHub Actions**, automatically deployed to **Hugging Face Spaces**.

> 🚀 **Live Demo**  
You can try the deployed model here:  
**https://huggingface.co/spaces/akincapar/Gender_Age_Race_Prediction**

> ⚠️ **Note:**  
The dataset is **not included in this repository**.  
You must provide the dataset in your environment before running the pipeline.

## 📊 About the Dataset

UTKFace dataset is a large-scale face dataset with long age span (range from 0 to 116 years old). The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc. This dataset could be used on a variety of tasks, e.g., face detection, age estimation, age progression/regression, landmark localization, etc.


## 📊 About the Dataset

The **UTKFace** dataset is a large-scale facial image collection spanning a wide age range (0–116 years).  
It contains **over 20,000 labeled face images**, each annotated with **age, gender, and ethnicity**.

The dataset includes significant variation in:
- Pose
- Facial expression
- Illumination
- Occlusion
- Image resolution

---

### ⚠️ Dataset Limitations & Real-World Challenges

Despite its popularity, UTKFace has several limitations that impact real-life model deployment:

1. **Annotation Noise**
   - Age, gender, and ethnicity labels are not always perfectly accurate.
   - Some age labels appear visually unrealistic, especially in extreme ranges (children, elderly).

2. **Ethnicity Imbalance**
   - Class distribution is skewed.
   - Certain demographic groups are significantly underrepresented, which can lead to biased model predictions.

3. **In-the-Wild ≠ Real-World**
   - Although images contain pose and lighting variations, they are still relatively curated compared to real-world production environments.
   - Models trained on UTKFace often achieve high validation metrics but degrade when used on:
     - camera streams,
     - unseen demographics,
     - low-quality/blurred photos,
     - or real-time input.

4. **Contextual Bias**
   - Many images are cropped close to the face, reducing noise.
   - Real-world applications must handle background, accessories (glasses, masks), motion blur, and environmental artifacts.

> **Summary:**  
Models trained on UTKFace may show strong results in training or test metrics, but **often struggle to generalize** in production use cases due to annotation noise, demographic imbalance, and domain shift.

📁 Dataset source:  
https://www.kaggle.com/datasets/jangedoo/utkface-new