import streamlit as st

# Sample dataset recommendations
dataset_recommendations = {
    "Weather": "weather.csv",
    "Stock Prices": "stocks.csv",
    # ... (other dataset recommendations)
}

# Define a function to answer questions
def answer_question(question):
    # Check if the question contains keywords for dataset recommendations
    keywords = dataset_recommendations.keys()
    for keyword in keywords:
        if keyword.lower() in question.lower():
            return f"I recommend the '{dataset_recommendations[keyword]}' dataset for {keyword}."

    # If not a recommendation question, provide a generic response
    return "I'm sorry, I don't have information about that dataset."

# Streamlit UI with enhanced design
def main():
    # Customizing the theme
    st.set_page_config(
        page_title=" 	:space_invader: Recomended Chatbot",
        page_icon="	:space_invader:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Adding a banner
    st.markdown(
        """
        <head> <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css" integrity="sha512-z3gLpd7yknf1YoNbCzqRKc4qyor8gaKU1qmn+CShxbuBusANI9QpRohGBreCFkKxLhei6S9CQXFEbbKuqLg0DA==" crossorigin="anonymous" referrerpolicy="no-referrer" /></head>
        <div style="background-color:#f0f0f0;padding:10px;border-radius:10px">
            <h1 style="color:#333">DataSet Recomender <i class="fa-solid fa-robot"></i> </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar with options
    st.sidebar.markdown("### Chatbot Options")

    # User input and submission button
    user_question = st.text_input("Ask me a question:")
    submit_button = st.button("Ask")

    if submit_button:
        response = answer_question(user_question)

        # Formatting the response
        st.markdown("### Chatbot's Response:")
        st.write(response)

if __name__ == "__main__":
    main()
