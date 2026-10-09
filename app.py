import io
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st
from reportlab.pdfgen import canvas
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title='InsureAI | College Demo', page_icon='🛡️', layout='wide')
st.markdown('''<style>.block-container{max-width:1150px;padding-top:2rem}.hero{background:linear-gradient(110deg,#142d59,#245bb0);padding:26px;border-radius:16px;color:white}.hero h1,.hero p{color:white;margin:4px 0}[data-testid="stMetric"]{background:#1b304e;padding:18px;border-radius:12px;border:1px solid #405573}[data-testid="stMetric"] *{color:white!important}</style>''',unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🛡️ InsureAI — College Demo</h1><p>Illustrative annual insurance premium calculator and ML analysis</p></div>',unsafe_allow_html=True)
st.warning('EDUCATIONAL DEMO ONLY: Premiums are generated using a hand-written illustrative pricing rule, not an insurer quote or a machine-learning prediction.')

# Illustrative rules deliberately constrained to ₹5,000–₹15,000.
def demo_premium(age, bmi, children, smoker):
    base = 5000
    age_component = max(0, age - 18) * 65
    bmi_component = max(0, bmi - 25) * 115
    child_component = children * 250
    smoker_component = 2500 if smoker == 'yes' else 0
    raw = base + age_component + bmi_component + child_component + smoker_component
    return round(min(15000, max(5000, raw)), 2)

@st.cache_data
def load_data():
    path = Path(__file__).with_name('insurance_working_dataset.csv')
    df = pd.read_csv(path)
    required = ['age','sex','bmi','children','smoker','region','charges']
    if any(c not in df.columns for c in required):
        raise ValueError('Dataset missing required columns')
    return df.dropna(subset=required)

@st.cache_resource
def train_ml(df):
    cols = ['age','sex','bmi','children','smoker','region']
    X_train,X_test,y_train,y_test = train_test_split(df[cols],df['charges'],test_size=.2,random_state=42)
    prep = ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),['sex','smoker','region']),('num','passthrough',['age','bmi','children'])])
    model = Pipeline([('pre',prep),('reg',GradientBoostingRegressor(random_state=42))])
    model.fit(X_train,y_train)
    predictions=model.predict(X_test)
    return {'MAE':mean_absolute_error(y_test,predictions),'RMSE':mean_squared_error(y_test,predictions)**.5,'R2':r2_score(y_test,predictions)}

def create_pdf(details, premium):
    buf=io.BytesIO()
    pdf=canvas.Canvas(buf)
    pdf.setTitle('InsureAI Educational Demo Report')
    pdf.setFont('Helvetica-Bold',18)
    pdf.drawString(42,790,'InsureAI - College Demo Report')
    pdf.setFont('Helvetica',11)
    pdf.drawString(42,765,datetime.now().strftime('%d %b %Y %I:%M %p'))
    y=720
    for key,value in details.items():
        pdf.drawString(42,y,f'{key}: {value}')
        y-=26
    pdf.setFont('Helvetica-Bold',14)
    pdf.drawString(42,y-30,f'Illustrative annual premium: INR {premium:,.2f}')
    pdf.setFont('Helvetica',10)
    pdf.drawString(42,y-65,'Educational rule-based illustration, capped at INR 15,000.')
    pdf.drawString(42,y-82,'NOT an insurer quote, medical risk score, or ML prediction.')
    pdf.save()
    return buf.getvalue()

tab1,tab2,tab3=st.tabs(['🎯 Demo Premium','📊 Demo Dashboard','🧠 ML Model Evaluation'])
with tab1:
    st.subheader('Customer details')
    with st.form('customer'):
        left,right=st.columns(2)
        with left:
            age=st.slider('Age',18,100,25)
            sex=st.selectbox('Sex',['male','female'])
            bmi=st.number_input('BMI',min_value=10.0,max_value=60.0,value=24.5,step=.1)
        with right:
            children=st.number_input('Children',min_value=0,max_value=10,value=0,step=1)
            smoker=st.selectbox('Smoker',['no','yes'])
            region=st.selectbox('Region',['northeast','northwest','southeast','southwest'])
        submit=st.form_submit_button('✨ Calculate Demo Premium',type='primary',use_container_width=True)
    if submit:
        premium=demo_premium(age,bmi,children,smoker)
        a,b,c=st.columns(3)
        a.metric('Illustrative Annual Premium',f'₹{premium:,.2f}')
        b.metric('Monthly Equivalent',f'₹{premium/12:,.2f}')
        c.metric('Demo Range','₹5,000 – ₹15,000')
        st.info('Rule: ₹5,000 base + age, BMI, children and smoking adjustments; final value capped at ₹15,000. Sex and region are collected for dataset compatibility but do not affect this demo rule.')
        details={'Age':age,'Sex':sex,'BMI':bmi,'Children':children,'Smoker':smoker,'Region':region}
        st.download_button('📄 Download Demo PDF Report',data=create_pdf(details,premium),file_name='insureai_demo_report.pdf',mime='application/pdf',use_container_width=True)
    else:
        st.info('Enter customer details and calculate the illustrative premium.')
with tab2:
    st.subheader('Illustrative pricing examples')
    examples=pd.DataFrame([{'Age':a,'BMI':b,'Children':c,'Smoker':s,'Demo annual premium (INR)':demo_premium(a,b,c,s)} for a,b,c,s in [(22,22,0,'no'),(30,26,1,'no'),(40,29,2,'yes'),(55,34,2,'yes'),(70,32,3,'yes')]])
    st.dataframe(examples,use_container_width=True,hide_index=True)
    st.bar_chart(examples.set_index('Age')['Demo annual premium (INR)'])
    st.caption('These are synthetic demonstration scenarios, not observed insurance prices.')
with tab3:
    st.subheader('Separate ML model — medical charges dataset')
    st.info('This Gradient Boosting model predicts charges from the original dataset in its original USD units. It does NOT produce the INR demo premium shown in the first tab.')
    try:
        df=load_data()
        metrics=train_ml(df)
        a,b,c=st.columns(3)
        a.metric('MAE (dataset USD)',f'${metrics["MAE"]:,.2f}')
        b.metric('RMSE (dataset USD)',f'${metrics["RMSE"]:,.2f}')
        c.metric('R² Score',f'{metrics["R2"]:.3f}')
        st.caption(f'80/20 train-test split • {len(df):,} dataset rows • Demonstration dataset metrics')
    except (FileNotFoundError,ValueError) as exc:
        st.error(f'ML dataset unavailable: {exc}')
        st.info('Place insurance_working_dataset.csv in the same folder as this app.')
st.divider()
st.caption('InsureAI • Educational demonstration only • No actual insurance quote or underwriting assessment')
