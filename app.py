import streamlit as st
from utils import auto_pipeline

st.title("Legal AI Summarizer & Simplifier")

input_text = st.text_area("Enter Legal Text", height=300)

if st.button("Summarize & Simplify"):
    if len(input_text) < 20:
        st.warning("Please enter a longer legal text!")
    else:
        with st.spinner("Working on it..."):
            summary, easy_summary = auto_pipeline(input_text)

        st.subheader("Generated Summary")
        st.success(summary)

        st.subheader("Easy Legal Summary")
        st.success(easy_summary)
