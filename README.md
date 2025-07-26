# Crop Yield Solutions

This repository provides practical Python scripts addressing key challenges in agricultural data science, with a focus on crop yield prediction and decision support. Each script demonstrates a solution to a common problem in the field, using synthetic or user-provided data.

## Table of Contents
- [Overview](#overview)
- [Programs](#programs)
  - [1. Data Quality and Availability](#1-data-quality-and-availability)
  - [2. Model Interpretability](#2-model-interpretability)
  - [3. Accessibility & Cost](#3-accessibility--cost)
  - [4. Optimal Harvest Scheduling](#4-optimal-harvest-scheduling)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [License](#license)

## Overview
This collection addresses:
- Data quality and missing values
- Model transparency and interpretability
- Cost-effective, accessible AI for farmers
- Data-driven harvest scheduling

Each script is self-contained and can be run independently.

## Programs

### 1. Data Quality and Availability ([1_data_availability.py](/1_data_availability.py))
**Problem:** Agricultural datasets often have missing values, noise, and limited size.

**Solution:**
- Imputes missing values using mean imputation
- Reduces noise with a median filter
- Demonstrates conceptual data augmentation for numerical features

**Usage:**
- Run as a script for a synthetic example, or pass your own DataFrame to `run_data_quality_solution()`

---

### 2. Model Interpretability ([2_model_interpretability.py](crop_yield_solutions/2_model_interpretability.py))
**Problem:** Complex AI models are often black boxes, making it hard to understand predictions.

**Solution:**
- Trains a Random Forest model for crop yield
- Uses SHAP (SHapley Additive exPlanations) to explain feature contributions
- Visualizes feature importance and individual predictions

**Usage:**
- Run as a script for a synthetic example, or pass your own DataFrame to `run_model_interpretability_solution()`

---

### 3. Accessibility & Cost ([3_cost_accessibility.py](crop_yield_solutions/3_cost_accessibility.py))
**Problem:** High costs and infrastructure gaps limit AI adoption in agriculture.

**Solution:**
- Demonstrates a lightweight Linear Regression model suitable for edge devices
- Simulates sending SMS alerts to farmers (conceptual, not actual SMS)

**Usage:**
- Run as a script for a synthetic example, or pass your own DataFrame and phone number to `run_accessibility_cost_solution()`

---

### 4. Optimal Harvest Scheduling ([4_optimal_harvest.py](crop_yield_solutions/4_optimal_harvest.py))
**Problem:** Harvest timing is often based on intuition, risking spoilage or lower quality.

**Solution:**
- Trains a Random Forest model to predict optimal harvest windows based on environmental and market data
- Visualizes predicted crop quality over time to identify the best harvest day

**Usage:**
- Run as a script for a synthetic example, or pass your own DataFrame and conditions to `run_optimal_harvest_scheduling_solution()`

---

## How to Run
1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd python_ds
   ```
2. Install the required Python packages (see below).
3. Run any script directly, e.g.:
   ```bash
   python crop_yield_solutions/1_data_availability.py
   ```
   Or import the functions in your own Python code.

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- scipy (for median filter in 1_data_availability.py)
- shap (for 2_model_interpretability.py)
- seaborn (for 4_optimal_harvest.py, optional)

Install all requirements with:
```bash
pip install pandas numpy scikit-learn matplotlib scipy shap seaborn
```

## License
This project is provided for educational and demonstration purposes. Please cite appropriately if used in research or derivative works. 
