import streamlit as st
import requests
import os
from PIL import Image

image_path = os.path.dirname(os.path.abspath(__file__)) + "/images/titanic_sinking.jpeg"

image = Image.open(image_path)
st.image(image, use_column_width=True)#, width=600)


'''
# Titanic survivor prediction app
'''

pclass = st.radio('Select the class of the passenger', (1, 2, 3))
gender = st.radio('Select the gender of the passenger', ("male", "female"))
age = st.number_input('Select age of the passenger', min_value=0, max_value=110, value=20)
fare = st.number_input('Select the ticket price the passenger paid', min_value=0.0, max_value=515.0, value=20.5)

url = f"https://titanic-n2o267u7cq-ew.a.run.app/survivor_predict?Sex={gender}&Age={age}&Fare={fare}&Pclass={pclass}"

response = requests.get(url).json()

if pclass == 1:
    pclass_textual = "first"
elif pclass == 2:
    pclass_textual = "second"
else:
    pclass_textual = "third"

if age <= 1:
    year = "year"
else:
    year = "years"

prob = str(round(response["Survived probability"]*100, 3)) + "%"
if response["Survived probability"] > 0.5:
    res = "Survived"
else:
    res = "Did not survive"

TITANIC_CSS = f"""
#proba_survived {{
    display: block;
    margin:auto;
    width:150px;
    flex-wrap: wrap;
    color:snow;
    font-size:50pt
}}
.recap {{
    color: green;
    font-weight:700;
}}
.result {{
    color: green;
    font-weight:700;
}}
#pred {{
    font-size:14pt;
    font-weight:700
}}
"""
PROBA_SURVIVED = f"""
<style>
{TITANIC_CSS}
</style>
<p>A <span class="recap">{age}</span> {year} old <span class="recap">{gender}</span> passenger who paid <span class="recap">{fare}$</span> for the ticket and who traveled in <span class="recap">{pclass_textual}</span> class would have a probability to survive of the Titanic sinking of:</p>
<div id="proba_survived">
    {prob}
</div>
<br/>
<div id="pred">
    Prediction: <span class="result">{res}</span>
</div>
"""
st.markdown("<body style='background-color: white;'>",unsafe_allow_html=True)
st.write(PROBA_SURVIVED, unsafe_allow_html=True)
