# Soil Fertility Prediction with Explainable AI

This repository contains the codebase for predicting soil fertility using supervised machine learning models, alongside the integration of Explainable AI (XAI) techniques to interpret the model's decision-making process.

This work is based on my published research in IEEE Access:
- 📄 [Explainable AI for Soil Fertility Prediction (IEEE Access 2023)](https://ieeexplore.ieee.org/document/10239153)

---

## Contents

- `data pipeline processing/`: Preprocessing scripts for the soil dataset
- `OriginalSoilDataset.csv`: Primary dataset used for training and evaluation
- `classifier_with_explaination.ipynb`: Main Jupyter Notebook with model training, prediction, and explanation visualization
- `datalabelsforparams.csv`: Feature label mappings
- `relativesoillabels.csv`: Label mappings for soil fertility classes

---

## Technologies Used

- Python 3
- Scikit-learn
- SHAP
- Jupyter Notebook

---

## Dataset

The dataset used for this project is publicly available on Kaggle:

- 📂 [Explainable AI for Soil Fertility Prediction Dataset](https://www.kaggle.com/datasets/harshivchandra/explainable-ai-for-soil-fertility-prediction)



### Dataset Description
The dataset includes soil sample records with the following types of features:
- **Chemical Properties**: Nitrogen, Phosphorus, Potassium levels, pH value, Organic Carbon content
- **Environmental Factors**: Temperature, Humidity, Rainfall
- **Fertility Labels**: Categorical soil fertility ratings

The data has been cleaned, normalized, and mapped into relative classes to enhance training performance for machine learning models.



### Dataset Files
- `OriginalSoilDataset.csv`: Raw feature data and fertility labels
- `relativesoillabels.csv`: Label mapping for fertility categories
- `datalabelsforparams.csv`: Feature name references for dataset columns



### License
This dataset is sourced from the LUCAS 2018 TopSoil Dataset, ESDAC, European Commission.

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/harshivchandra/Soil-Fertility-Prediction-with-Explainable-AI.git
cd Soil-Fertility-Prediction-with-Explainable-AI

```

2. Install requirements (via requirements.txt):

```bash
pip install -r requirements.txt
```

3. Run notebook in terminal

```bash
classifier_with_explaination.ipynb
```
---

## Future Work

- Add hyperparameter tuning for classifiers
- Extend XAI support with additional explanation methods
- Deploy model using a lightweight web app for predictions
- Improve preprocessing and feature engineering pipelines
---

## Citation

If you use or refer to this work, please cite:
```
@article{chandra2023explainable,
  title={Explainable AI for Soil Fertility Prediction},
  author={Chandra, Harshiv and Pawar, Pranav M. and Elakkiya, R. and Tamizharasan, P. S. and Muthalagu, Raja and Panthakkan, Alavikunhu},
  journal={IEEE Access},
  volume={11},
  pages={97866--97878},
  year={2023},
  publisher={IEEE}
}
```



