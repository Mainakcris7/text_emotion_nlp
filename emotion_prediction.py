import streamlit as st
import spacy
import joblib
import numpy as np
from cleantext import clean
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_resource
def load_data_model():
    tfidf_vect = joblib.load("./tfidf_vect.pkl")
    catboost_model = joblib.load("./catboost_model.pkl")
    return tfidf_vect, catboost_model


# @st.cache_data
def clean_text(text, tfidf):
    nlp = spacy.load('en_core_web_sm')
    text = clean(text, no_urls=True, no_emails=True, no_numbers=True,
                 no_currency_symbols=True, no_punct=True, no_emoji=True,
                 replace_with_url='', replace_with_number='', replace_with_currency_symbol='',
                 replace_with_email='', no_line_breaks=True)
    doc = nlp(text)
    text = [token.lemma_ for token in doc
            if not token.is_stop and len(token.text) > 1 and token.lemma_ not in ['feel', 'feeling', 've']]
    text = " ".join(text)
    text_vect = tfidf.transform([text])
    return text_vect


# @st.cache_data
def get_prediction(_model, _text_vect):
    label = ['anger', 'fear', 'joy']
    pred_proba = _model.predict_proba(_text_vect)
    index = int(np.argmax(pred_proba, axis=1))
    return label[index], pred_proba


st.title("Text emotion predictor 😡😨😃")
text = st.text_area("Enter your text here")

clicked = st.button("Predict")
if text and clicked:
    tfidf_vect, model = load_data_model()
    text_vect = clean_text(text, tfidf_vect)
    pred_label, pred_proba = get_prediction(model, text_vect)
    if pred_label == 'joy':
        st.write(f"## Predicted emotion: :green[{pred_label}]")
    elif pred_label == 'anger':
        st.write(f"## Predicted emotion: :red[{pred_label}]")
    else:
        st.write(f"## Predicted emotion: :blue[{pred_label}]")

    col1, col2 = st.columns(2)
    with col1:
        fig = plt.figure(figsize=(4, 1.5))
        plt.style.use('dark_background')
        sns.barplot(x=pred_proba[0], y=['anger', 'fear', 'joy'], palette=[
                    '#f72d3d', '#497cfc', '#02eb6f'])
        plt.xticks(())
        plt.yticks(fontweight='bold')
        for i, prob in enumerate(pred_proba[0]):
            if prob < 0.2:
                plt.text(prob + 0.012, i+0.1,
                         np.round(prob, 3), fontweight='bold')
            else:
                plt.text(prob - 0.14, i+0.1,
                         np.round(prob, 3), fontweight='bold', color='black')

        sns.despine()
        st.pyplot(fig)
