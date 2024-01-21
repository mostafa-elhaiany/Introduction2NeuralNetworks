import streamlit as st
from Content import ActivationFunctions, NetworkArchitecture, Dataset

def main():
    st.title("Introduction to Neural Networks")
    st.write("This Streamlit app is designed to be an educational and hands-on tool for individuals interested in Neural Networks. It provides an intuitive interface to experiment with mathematical functions, generate datasets, and observe how neural networks learn from and predict on different data patterns.")
    st.divider()

    ActivationFunctions.draw(st)
    st.divider()
    

    NetworkArchitecture.draw(st)
    st.divider()

    Dataset.draw(st)
    st.divider()

if __name__ == "__main__":
    main()