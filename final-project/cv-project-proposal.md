# Computer Vision Final Project Proposal: GeoGuessr AI using DINOv3

**Student:** Tsung-Wei (Miles) Chin, Yu-Fang (Brenda) Lin
**Date:** November 3, 2025

---

## 1. Problem Statement

This project aims to train a lightweight prediction head on top of the **DINOv3** vision transformer to predict geographic coordinates (latitude and longitude) from street-view images for automated GeoGuessr gameplay. DINOv3 is the latest large-scale vision transformer that has demonstrated state-of-the-art performance across various vision tasks, often surpassing many fine-tuned and domain-specific models. This project investigates whether DINOv3's powerful learned representations can compete with specialized geolocation models like **PIGEON**, which was specifically designed for this task and achieved top 0.01% human player performance.

The core research question is: **Can a general-purpose vision foundation model (DINOv3) match or exceed specialized geolocation models when trained with a lightweight prediction head?**

---

## 2. Problem Relevance

### Practical Applications
While geolocation from images is entertaining in the context of GeoGuessr gameplay, it has significant real-world applications:

- **Crime Investigation**: Identifying locations from surveillance footage, social media images, or evidence photos
- **Search and Rescue**: Locating individuals from photos or videos when GPS data is unavailable
- **Digital Forensics**: Verifying the authenticity and origin of digital content
- **Content Moderation**: Identifying geographic origins of problematic online content
- **Wildlife Conservation**: Tracking animal locations from camera trap images

### Academic Interest
This project also explores the **transfer learning capabilities** of large foundation models, testing whether general-purpose visual representations can be efficiently adapted to specialized tasks with minimal additional training.

---

## 3. Prior Art and Related Work

### Existing Geolocation Models

**PIGEON (Stanford, 2024)**: The current state-of-the-art geolocation model achieved 92% country-level accuracy and >40% accuracy within 25km globally. It uses semantic geocell creation and multi-task contrastive pretraining, ranking in the top 0.01% of human GeoGuessr players.

**OpenAI's o3 (2025)**: Recently achieved superhuman performance with 23,179 points vs. Master I-ranked human's 22,054 points by combining vision capabilities with chain-of-thought reasoning.

**GeoCLIP (2023)**: Introduced hierarchical location encoding using random Fourier features, achieving 74.1% continent-level accuracy through CLIP-inspired alignment between locations and images.

**GAEA (2025)**: A conversational geolocation model achieving 66.06% average accuracy with conversational reasoning capabilities.

### Foundation Models

**DINOv3**: A self-supervised vision transformer trained on diverse visual data that has demonstrated exceptional transfer learning performance across multiple downstream tasks without task-specific fine-tuning.

---

## 4. Proposed Method

### Model Architecture

The proposed approach leverages DINOv3's frozen pre-trained backbone with a trainable prediction head:

1. **Feature Extraction**: Use pre-trained **DINOv3-ViT** (e.g., ViT-L/14 or ViT-g/14) as a frozen feature extractor
2. **Prediction Head**: Train a lightweight head on top, with two architecture options:
   - **Option A**: Simple linear layer(s) for direct coordinate regression
   - **Option B**: Additional transformer layers for spatial reasoning, following DINOv3's text-alignment example

### Training Strategy

**Dataset**: OpenStreetView-5M (OSV5M) dataset or Google Street View datasets from Kaggle
- OSV5M contains 5.1M geo-referenced street view images from 225 countries
- Alternative: Start with smaller Kaggle datasets (10,000-25,000 images) for prototyping

**Training Approach**:
- Freeze DINOv3 backbone weights to leverage pre-trained representations
- Train only the prediction head to minimize computational requirements
- Use geocell classification (similar to PIGEON) rather than direct regression to avoid "landing in oceans"
- Alternatively, explore hybrid approach: coarse classification + fine coordinate regression

<!--
**Loss Function**: 
- Haversine distance loss for coordinate regression
- Cross-entropy loss for geocell classification
- Potentially combine both approaches hierarchically
-->

<!--
### Evaluation Metrics

1. **Distance-based accuracy**: Percentage of predictions within 1km, 25km, 200km, 750km, 2500km
2. **Country-level accuracy**: Correct country prediction rate
3. **GeoGuessr score**: Standard 5000-point scoring system
4. **Median and mean error distance**: In kilometers
5. **Comparison baseline**: PIGEON performance on same test set
-->

### Expected Outcomes

- **Success**: Achieve country-level accuracy >80% and demonstrate that DINOv3's representations are effective for geolocation
- **Baseline**: Match or approach PIGEON's performance

---

<!--
## 5. Implementation Plan

### Phase 1: Setup and Baseline (Week 1)
- Download and preprocess OSV5M dataset or Kaggle Street View dataset
- Set up DINOv3 feature extraction pipeline
- Implement simple linear head baseline

### Phase 2: Model Development (Week 2-3)
- Experiment with different head architectures (linear vs. transformer layers)
- Implement geocell classification approach
- Train and validate initial models

### Phase 3: Evaluation and Analysis (Week 4)
- Comprehensive evaluation using multiple metrics
- Compare performance against PIGEON baseline
- Analyze failure cases and geographical biases
- Prepare final report and presentation

---

## 6. Required Resources

- **Computational**: GPU with 16GB+ VRAM (for inference and head training)
- **Data**: OSV5M dataset (~100GB) or Kaggle Street View datasets (~1-5GB)
- **Software**: PyTorch, Hugging Face Transformers, DINOv3 pre-trained weights
- **Storage**: 100-200GB for dataset and model checkpoints

---

## 7. Expected Deliverables

1. Trained geolocation model with DINOv3 backbone
2. Comprehensive evaluation report comparing performance to PIGEON
3. Analysis of DINOv3's effectiveness for geographic reasoning
4. Code repository with training and inference scripts
5. Final presentation demonstrating model capabilities

---
-->

## References

1. OpenStreetView-5M: The Many Roads to Global Visual Geolocation. CVPR 2024.
2. PIGEON: Predicting Image Geolocations. CVPR 2024.
3. GeoCLIP: Clip-Inspired Alignment between Locations and Images. NeurIPS 2023.
4. GAEA: A Geolocation Aware Conversational Model. 2025.
5. DINOv3: A Self-supervised Vision Transformer. Meta AI Research.
6. GSV-Cities: Toward Appropriate Supervised Visual Place Recognition. 2022.
7. Mapillary Street-Level Sequences Dataset. CVPR 2020.