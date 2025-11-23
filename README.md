# Biological Age Prediction

A machine learning framework for predicting medical outcomes including biological age and mortality risk using Opportunistic Cardiometabolic Screening data. This project combines CT imaging data with clinical variables to build predictive models for healthcare applications.

## Problem Statement

Chronological age doesn't always reflect an individual's true physiological state. **Biological age**—a measure of how "old" a person's body functions—can differ significantly from chronological age and is a better predictor of health outcomes. This project addresses:

- **Biological Age Estimation**: Predicting physiological age from medical imaging and clinical data
- **Mortality Risk Prediction**: Forecasting death outcomes using multi-modal health data
- **Multi-Modal Learning**: Combining CT imaging features with clinical variables for improved accuracy

## Features

- **Multi-Modal Input**: Integrates CT scan data with clinical variables
- **Multiple Prediction Tasks**: 
  - Biological age regression
  - Mortality risk classification
  - Additional medical outcome predictions
- **Neural Network Architectures**: Custom models for different prediction tasks
- **Data Augmentation**: Enhanced training data for improved model generalization
- **Comprehensive Evaluation**: Low error and high accuracy metrics

## Project Structure

```
Biological-Age-Prediction/
├── Train Src/              # Training scripts for different tasks
│   ├── part1.py           # Task 1: Biological age prediction
│   ├── part2.py           # Task 2: Additional outcome prediction
│   └── part3.py           # Task 3: Mortality/death prediction
├── Age_Vs_BioAge.py       # Analysis and visualization
├── bio_age_nn.py          # Neural network for biological age
├── data/                  # Dataset files (CT and clinical data)
├── models/                # Saved model checkpoints
└── requirements.txt       # Python dependencies
```

## Technical Details

### Data Sources

- **CT Imaging Data**: Opportunistic Cardiometabolic Screening CT scans
- **Clinical Variables**: Patient demographics, lab values, and health metrics
- **Outcomes**: Biological age, mortality indicators, and other medical endpoints

### Model Architecture

- **Neural Networks**: Custom architectures for regression and classification
- **Feature Engineering**: Extraction of relevant features from CT and clinical data
- **Data Preprocessing**: Normalization, augmentation, and train/test splitting

## Quick Start

### Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Training

Training is divided into three parts corresponding to different prediction tasks:

```bash
cd "Train Src"
python part1.py  # Biological age prediction
python part2.py  # Additional medical outcome
python part3.py  # Mortality/death prediction
```

### Demo

Run the driver script to test on sample CT data:

```bash
python Age_Vs_BioAge.py
```

## Results

The models achieve:
- **Low Error**: Accurate biological age predictions
- **High Accuracy**: Reliable classification of medical outcomes
- **Clinical Relevance**: Predictions that can inform healthcare decisions

## Applications

- **Preventive Medicine**: Identifying patients at higher biological age
- **Risk Stratification**: Assessing mortality and health risks
- **Research**: Understanding factors that influence biological aging

## License

See repository for license information.

