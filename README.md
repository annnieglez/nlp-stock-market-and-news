# 🧠 Fake News Detection in News Headlines – NLP, Sentiment, & Embeddings

## Overview

This project is an NLP-driven system designed to detect fake news from news headlines. It combines sentiment analysis and news embeddings generated using transformer models from Hugging Face to power a binary classification model. 

## Features

- **Sentiment Analysis**: Extracts the emotional tone of news headlines to identify potential bias.

- **Diverse Embedding Techniques**: Supports multiple methods to convert headlines into vector representations:
  - **TF-IDF**: Highlights important terms based on term frequency and inverse document frequency.
  - **Bag of Words**: Simple frequency-based vectorization method.
  - **Word2Vec**: Captures semantic relationships between words using shallow neural networks.
  - **BERT (`bert-base-uncased`)**: Generates contextual embeddings using transformer architecture.
  - **MiniLM**: Lightweight transformer model for efficient embedding with strong performance.

- **SVC Classification Model**: A Support Vector Classifier trained on embeddings to distinguish between real and fake news headlines.

- **Model Evaluation**: Performance is assessed using standard classification metrics (accuracy, F1-score, confusion matrix).

## Key Steps

1. **Data Cleaning** and **Exploratory Data Analysis (EDA)**  

2. **Sentiment Analysis**  
   - Applied **VADER** for rule-based sentiment scoring.
   - Used **RoBERTa** for contextual, transformer-based sentiment classification.

3. **Text Embedding Generation**  
   - Transformed news headlines into numerical vectors using multiple techniques:
     - **TF-IDF**
     - **Bag of Words**
     - **Word2Vec**
     - **BERT (`bert-base-uncased`)**
     - **MiniLM**

4. **Model Comparison**  
   - Tested several machine learning models using different combinations of embeddings and sentiment scores.

5. **Final Model Training**  
   - Trained a **Support Vector Classifier (SVC)** using **BERT embeddings**.
   - Achieved an accuracy of **0.9627**.
   - **Cross-validation** confirmed model robustness and generalization.

## Project Structure

Below is the project structure with a brief explanation of each component:

```bash
nlp-stock-market-and-news
├── data                           # Datasets directory
│ └── data_to_clean                # Original data before preprocessing
├── images                         # Visual assets like confusion matrix, ROC curve, etc.
├── nlp_scripts                    # Python scripts for data cleaning, EDA and NLP tasks embedding and modeling
├── notebooks                      # Jupyter notebooks for EDA, data cleaning and modeling
├── report                         # Folder for storing final reports and visualizations
├── .gitattributes                 # Git attributes (e.g., Git LFS for large files like CSVs)
├── .gitignore                     # Specifies untracked files to ignore
├── LICENSE                        # License for use, modification, and distribution
├── README.md                      # Project overview, setup instructions, and key details
├── requirements.txt               # List of Python dependencies
├── pyproject.toml                 # Configuration file for project setup
├── setup.py                       # Setup script for packaging the project
└── update_vscode.py               # Script to configure VS Code settings for this project
```

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/annnieglez/nlp-stock-market-and-news
cd nlp-stock-market-and-news
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv nlp_news
source nlp_news/bin/activate        # On macOS/Linux
source nlp_news\Scripts\activate    # On Windows
```

### 3. Update VSCode Settings (Optional)

Run the following script to configure VSCode for the project:

```bash
python update_vscode.py
```

### 4. Install the Project in Editable Mode

From the main directory, run:

```bash
pip install -e .
```

to install the custom scripts.

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the custom notebooks

The `notebooks` directory contains Jupyter notebooks for data cleaning, EDA, sentiment analysis, embedding generation, and model training. To run them:

1. Open your preferred Jupyter environment (JupyterLab, Jupyter Notebook, or VS Code).
2. Navigate to the `notebooks` folder.
3. Open the notebook you want to run (e.g., `data_cleaning.ipynb`, `eda.ipynb`, `nlp.ipynb`).
4. Execute the cells in order, following the instructions and comments provided in each notebook.
5. Review the outputs, visualizations, and results as you progress through each step.

### 7. Running in Google Colab

The code for RoBERTa sentimenst and emebedings and for the fake news model was developed in Google Colab since embeddings processing is computationally intensive. To run the notebooks, follow these steps:

1. Open the Google Colab notebooks:
    - Upload the provided notebook files.
2. Mount your Google Drive:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
3. Copy the VADER datasets to your Google Drive.
4. Update the `folder_path` variable to match the location of your dataset folder in your Google Drive or local system.
5. Run the `roberta_sentiments_and_embedings` and `fake_news_model`.

## Future Improvements: Stock Market Prediction from News Headlines

A promising extension of this project is to leverage news headlines for predicting stock market movements. The goal is to build models that use news sentiment and embeddings to forecast whether a stock's closing price will decrease.

### Data Collection

A dedicated notebook is already provided to:
- Collect historical stock data from Yahoo Finance.
- Aggregate news headlines from sources such as Google News and Markets Insider.

### Data Preparation

- Data cleaning for both stock and news datasets has been completed, following the same procedures as for the fake news dataset.
- Embeddings and sentiment analysis are also prepared using the established pipeline.

### Next Steps

- Develop a predictive model that uses the generated embeddings and sentiment scores from news headlines to classify whether a stock's closing price will decrease.
- Experiment with different modeling approaches (e.g., logistic regression, SVC, or neural networks) and evaluate performance using appropriate metrics.

This enhancement will enable the project to not only detect fake news but also explore the relationship between news sentiment and stock market behavior.

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and create a new branch for your changes.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the `LICENSE` file for details.

## Author

This project was created by Annie Meneses Gonzalez. Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/annie-meneses-gonzalez-57bb9b145/).

