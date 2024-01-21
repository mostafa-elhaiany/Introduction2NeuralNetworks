from Utils.DrawNN import DrawNN
import re

def draw(st):
    collect_numbers = lambda x : [min(32,int(i)) for i in re.split("[^0-9]", x) if i != ""]
    st.header("Neural network architectures")
    st.write("Let's learn how the architecture looks like and create some different ones together. (Numbers capped at 32 to avoid overloading the server)")
    input_text = st.text_input("write network arcitecture, example = [1,2,2,1]")
    if(input_text):
        numbers = collect_numbers(input_text)
        st.write(f"using {numbers} as the list of numbers")
        try:
            network = DrawNN(numbers)
            figure = network.draw()
            st.pyplot(figure,use_container_width=False)
        except Exception as e:
            print(e)
            st.write("Please use a correct notation")
    else:
        pass
    