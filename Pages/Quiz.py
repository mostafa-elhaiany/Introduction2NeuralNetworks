import streamlit as st
from Content import QuizTraining
def main():
    st.title("Neural Networks Quiz")
    st.write("In this page we will test your knowledge of neural networks.")
    st.write("Your Task is to create a model that gets the best accuracy on the following dataset.")

    QuizTraining.draw(st)

if __name__ == "__main__":
    main()