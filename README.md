# Spotify Popularity Prediction Using Machine Learning

**An End-to-End Machine Learning Regression Project**

## 1. Project Overview

Spotify Popularity Prediction is a Machine Learning project that aims to estimate the popularity score of a music track based on its audio characteristics, genre, artist information, and album details.

The project uses historical Spotify track data to identify patterns associated with music popularity and develops a regression model to predict popularity scores on a scale of 0 to 100.

The complete workflow includes data preprocessing, exploratory data analysis, feature engineering, model comparison, model evaluation, and an interactive web application developed using Streamlit.

## 2. Project Objectives

- Analyze Spotify track data to understand the factors associated with music popularity.
- Perform data cleaning and exploratory data analysis (EDA).
- Engineer meaningful features to improve model performance.
- Apply appropriate preprocessing techniques to numerical and categorical variables.
- Compare multiple Machine Learning regression algorithms.
- Develop and evaluate a Random Forest Regression model.
- Build an interactive application for generating popularity predictions.

## 3. Dataset Description

**Dataset:** Spotify Tracks Dataset  
**Source:** Kaggle  
**Original Dataset Size:** 114,000 records  
**Original Features:** 21 columns

The dataset contains track-level information, including:

| Feature | Description |
|---|---|
| `track_id` | Spotify track identifier |
| `track_name` | Name of the track |
| `artists` | Artist or artists associated with the track |
| `album_name` | Album containing the track |
| `track_genre` | Genre classification |
| `popularity` | Track popularity score (0–100) |
| `duration_ms` | Track duration in milliseconds |
| `danceability` | Suitability of a track for dancing |
| `energy` | Perceived intensity and activity |
| `loudness` | Overall loudness in decibels |
| `speechiness` | Presence of spoken words |
| `acousticness` | Likelihood that a track is acoustic |
| `instrumentalness` | Likelihood that a track contains no vocals |
| `liveness` | Likelihood of a live performance recording |
| `valence` | Musical positivity conveyed by a track |
| `tempo` | Estimated tempo in beats per minute |
| `key` | Musical key |
| `mode` | Major or minor mode |
| `time_signature` | Estimated beats per bar |

The dataset is used for regression, with `popularity` as the target variable.

## 4. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model Serialization | Joblib |
| Web Application | Streamlit |
| Development Environment | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |

## 5. Project Workflow

### 5.1 Data Loading and Exploration

- Loaded the dataset using Pandas.
- Examined the dataset structure, dimensions, and data types.
- Analyzed numerical feature distributions using descriptive statistics.
- Investigated missing values and duplicate track identifiers.

### 5.2 Exploratory Data Analysis

Performed EDA to understand the underlying characteristics of the dataset.

Key analysis areas included:

- Distribution of Spotify popularity scores.
- Distribution of audio features.
- Genre distribution.
- Correlation analysis between numerical features.
- Identification of missing values and duplicate identifiers.
- Investigation of relationships between track characteristics and popularity.

### 5.3 Data Cleaning

The following cleaning operations were performed:

- Removed the unnecessary `Unnamed: 0` column.
- Identified and removed one invalid record with zero track duration.
- Investigated missing values in numerical features.
- Examined duplicate track IDs without automatically treating different records with the same ID as identical.

After cleaning, the dataset contained **113,999 records and 21 columns**.

### 5.4 Feature Engineering

Created additional features to capture information that was not directly represented by the original dataset.

| Engineered Feature | Description |
|---|---|
| `artist_count` | Number of artists associated with a track |
| `duration_minutes` | Track duration converted from milliseconds to minutes |
| `artist_track_count` | Frequency of an artist in the training dataset |
| `album_track_count` | Frequency of an album in the training dataset |

Artist and album frequency features were calculated using training data only to prevent information leakage from the test dataset.

### 5.5 Data Splitting and Leakage Prevention

A group-aware train-test split was performed using `track_id` to prevent records belonging to the same track identifier from being distributed across training and testing datasets.

| Dataset | Records |
|---|---:|
| Training Set | 91,281 |
| Testing Set | 22,718 |

Training-only frequency mappings were also used when creating artist and album frequency features.

Previously unseen artists and albums in the test dataset were assigned a frequency value of zero.

### 5.6 Data Preprocessing

A ColumnTransformer was implemented to handle numerical and categorical features separately.

**Numerical preprocessing:**
- SimpleImputer using the median strategy.
- StandardScaler for feature scaling.

**Categorical preprocessing:**
- SimpleImputer using the most-frequent strategy.
- OneHotEncoder with `handle_unknown="ignore"`.

The preprocessing pipeline transformed the original 19 model input features into **132 processed features**.

### 5.7 Model Development

Multiple regression algorithms were evaluated to identify a suitable model for the prediction task.

Algorithms considered:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Extra Trees Regressor

Based on the preliminary model comparison, Random Forest Regression was selected for the final implementation.

### 5.8 Random Forest Regression

Random Forest is an ensemble learning algorithm that combines multiple decision trees to generate predictions.

For regression, the final prediction is obtained by averaging the predictions of individual trees.

The selected model configuration was:

| Hyperparameter | Value |
|---|---:|
| `n_estimators` | 50 |
| `max_depth` | 20 |
| `max_features` | sqrt |
| `min_samples_split` | 10 |
| `min_samples_leaf` | 5 |
| `random_state` | 42 |
| `n_jobs` | -1 |

These settings control the number of trees, tree complexity, feature selection, and minimum sample requirements for tree splitting.

## 6. Model Evaluation

The final Random Forest Regression model was evaluated using three standard regression metrics.

### Test Set Performance

| Evaluation Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 12.66 |
| Root Mean Squared Error (RMSE) | 15.79 |
| R² Score | 0.4998 |

### Metric Interpretation

**Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted popularity scores. The model achieved an MAE of approximately 12.66 popularity points.

**Root Mean Squared Error (RMSE):** Measures prediction error while giving greater weight to larger errors. The model achieved an RMSE of approximately 15.79.

**R² Score:** Measures the proportion of variation in the target variable explained by the model. The model achieved an R² score of approximately 0.50 on the test dataset.

### Cross-Validation

Five-fold GroupKFold cross-validation was performed on the training dataset to evaluate model consistency while respecting track ID groups.

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| MAE | 12.487 | 0.100 |
| RMSE | 15.616 | 0.096 |
| R² Score | 0.5094 | 0.0045 |

## 7. Model Serialization

The trained Machine Learning components were saved using Joblib to enable reuse without retraining the model whenever the application starts.

The following artifacts were generated:

- `spotify_random_forest.pkl` – Trained Random Forest Regression model.
- `spotify_preprocessor.pkl` – Fitted preprocessing pipeline.
- `spotify_feature_mappings.pkl` – Training-derived artist and album frequency mappings.

These artifacts are loaded by the Streamlit application to process user inputs and generate predictions.

## 8. Streamlit Application

An interactive web application was developed using Streamlit to make the trained model accessible through a user-friendly interface.

### Application Features

- Artist and album information input.
- Music genre selection.
- Audio feature configuration.
- Popularity prediction using the trained model.
- Low, Medium, and High popularity category display.
- Prediction history within the current application session.
- Interactive dashboard with a dark-themed user interface.

### Prediction Workflow

1. The user enters the track information.
2. The application creates the required input features.
3. Feature engineering is performed.
4. The saved preprocessing pipeline transforms the input.
5. The Random Forest model generates the popularity prediction.
6. The predicted score and corresponding category are displayed.

The predicted score is constrained to the 0–100 range for display.

## 9. Project Structure

```text
Music-Recommendation-ML-Project/
│
├── APP/
│   └── app.py
│
├── MODEL/
│   ├── spotify_random_forest.pkl
│   ├── spotify_preprocessor.pkl
│   └── spotify_feature_mappings.pkl
│
├── NOTEBOOK_v2/
│   ├── music_recommendation.ipynb
│   └── dataset.csv
│
├── .gitignore
└── README.md
```

**Note:** The dataset and serialized model artifacts are excluded from Git tracking. They must be generated or supplied separately before running the application.

## 10. Installation and Execution

### Prerequisites

- Python 3.10 or a compatible Python environment.
- Git.
- Required Python libraries.

### Step 1: Clone the repository

```bash
git clone https://github.com/aswathy325/Music-Recommendation-ML-Project.git
```

### Step 2: Navigate to the project directory

```bash
cd Music-Recommendation-ML-Project
```

### Step 3: Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

### Step 4: Prepare the dataset

Download the Spotify Tracks Dataset from Kaggle and place the CSV file in the `NOTEBOOK_v2` directory.

### Step 5: Generate the model artifacts

Run the notebook's training and model-saving workflow to generate the required files in the `MODEL` directory.

### Step 6: Launch the application

```bash
streamlit run APP/app.py
```

The Streamlit application will open in your default web browser.

## 11. Limitations

- Spotify popularity is influenced by external factors such as marketing, playlist placement, release
