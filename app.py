import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

st.title("🎯 Customer Intent Classification App")

choice = st.selectbox("Choose Model:", ["TF-IDF", "BiLSTM (TFLite)", "TinyBERT"])
text = st.text_area("Enter customer message:")

if st.button("Predict") and text.strip() != "":
    if choice == "TF-IDF":
        tfidf = joblib.load("models/tfidf/tfidf_vectorizer.joblib")
        clf = joblib.load("models/tfidf/tfidf_classifier.joblib")
        lb = joblib.load("models/tfidf/label_binarizer.joblib")
        x = tfidf.transform([text])
        pred = clf.predict(x)
        st.success(lb.inverse_transform(pred)[0][0])

    elif choice == "BiLSTM (TFLite)":
        interpreter = tf.lite.Interpreter(model_path="models/bilstm/bilstm_model_fixed.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        tokenizer = joblib.load("models/bilstm/tokenizer_bilstm.joblib")
        label_enc = joblib.load("models/bilstm/label_encoder.joblib")

        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=100)

        interpreter.set_tensor(input_details[0]['index'], padded.astype(np.int32))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        st.success(label_enc.inverse_transform([np.argmax(output)])[0])

    elif choice == "TinyBERT":
        tokenizer = AutoTokenizer.from_pretrained("models/tinybert/tinybert_model")
        model = AutoModelForSequenceClassification.from_pretrained("models/tinybert/tinybert_model")
        label_enc = joblib.load("models/tinybert/label_encoder.joblib")

        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        logits = model(**tokens).logits
        pred = torch.argmax(logits, dim=1).item()
        st.success(label_enc.inverse_transform([pred])[0])
