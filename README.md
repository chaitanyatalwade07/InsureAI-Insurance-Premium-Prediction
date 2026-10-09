# AI-Based Insurance Premium Prediction

Student: Chaitanya Talwade
PRN: 1022411058

## Included
- insurance_working_dataset.csv
- insurance_premium_prediction.py
- Insurance_Premium_Prediction.ipynb
- Insurance_Premium_Prediction_Final_Report.docx
- Insurance_Premium_Prediction_Final_Presentation.pptx
- app.py
- requirements.txt
- model_results.csv
- sample_prediction.txt
- charts/

## Dataset source
The project follows the public Medical Cost Personal Dataset schema:
https://www.kaggle.com/datasets/mirichoi0218/insurance
GitHub copy:
https://github.com/krish1407/Medical-Cost-Personal-Datasets/blob/master/insurance.csv

## Important dataset note
The included CSV is a generated working/demo dataset with the same columns and 1,338-row size as the common public dataset. It is included so the project runs offline. If your teacher requires the original public dataset, download the original `insurance.csv`, replace the included working CSV, rerun the notebook/script, and update the report/PPT metrics.

## Run
pip install -r requirements.txt
python insurance_premium_prediction.py

## Streamlit
streamlit run app.py
