import streamlit as st
from Content import Training, Dataset

def main():
    st.title("Neural Networks Training")
    st.write("In this page we will use the knowledge we learnt to train a machine learning model")

    X, y, is_regression = Dataset.draw(st, 1, 56)

    Training.draw(st,is_regression,X,y)

if __name__ == "__main__":
    main()