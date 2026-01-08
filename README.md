# Music Genre Classification

## Description
This project explores the application of machine learning algorithms to classify audio tracks into 10 distinct musical genres (Blues, Classical, Country, Disco, Hip Hop, Jazz, Metal, Pop, Reggae, Rock). Using the **GTZAN dataset**, the project involves feature extraction, data visualization, and a comparative analysis of three different classification models.

## Technology Stack
* **Language:** Python
* **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
* **Dataset:** GTZAN (1,000 audio tracks, 10 genres) 

## Key Features
* **Feature Selection:** Analyzed spectral features including **MFCCs**, **Spectral Centroid**, **Zero Crossing Rate**, and **Spectral Bandwidth** to distinguish between genres.
* **Data Visualization:** Created correlation heatmaps and feature variance plots to identify the most discriminative audio characteristics.
* **Model Comparison:** Implemented and evaluated three supervised learning models:
    1.  **K-Nearest Neighbors (KNN):** A simple, instance-based learner.
    2.  **Logistic Regression:** A linear model used as a baseline.
    3.  **Random Forest:** An ensemble method that handles non-linear relationships.

## Results
The models were evaluated on a held-out test set (20% of data). **Random Forest** proved to be the most effective model, achieving the highest accuracy and robustness against overfitting.

| Model | Test Accuracy | Observations |
| :--- | :--- | :--- |
| **Random Forest** | **68%** | Best performer. [cite_start]Effectively handled complex feature relationships. |
| **KNN** | 60% | [cite_start]Improved with dataset size but struggled with high dimensions. |
| **Logistic Regression** | 50% | [cite_start]Lowest performance, indicating non-linear boundaries between genres. |

## Acknowledgments
This project was developed as part of the course **ELEC-C5220** at Aalto University.
* **Core Implementation:** The contents of the `src` directory (Model, Trainer, Signal Processing) represent my personal contribution.
* **Framework:** The testing suite and project scaffolding were provided by the course instructors.
