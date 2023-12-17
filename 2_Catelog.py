import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Catalog", page_icon=" :thought_balloon:", layout="wide")

# Header Section
st.title("	:milky_way: SPACE SEARCH!!")

# Function to display dataset and add statistical table and download button
def display_dataset(data_path, title, description, image_path):
    st.write(f"---")
    st.write(f"# {title}")
    
    # Create a container for the dataset display
    dataset_container = st.container()
    
    with dataset_container:
        data = pd.read_csv(data_path)
        
        st.write("### Dataset")
        st.dataframe(data)
        
        # Add a statistical table
        st.write("### Statistical Summary")
        st.table(data.describe())
        
        # Add a download button for the dataset
        csv_file = data.to_csv(index=False)
        st.download_button(label=f"Download {title} Dataset", data=csv_file, file_name=f"{title}.csv")
    
    # Catalog information and image
    st.write(f"### Catalog")
    st.write(description)
    st.image(image_path, use_column_width=True)

# Display datasets
display_dataset('D:\\Space\\multipleapp_pages\\csv\\Tree.csv', "Trees", "A tree is a tall plant...", "./assetes/images/tree1.jpg")
display_dataset('D:\\Space\\multipleapp_pages\\csv\\AIR.csv', "Air-Quality", "A tree is a tall plant...", "./assetes/images/Air.jpg")
display_dataset('D:\\Space\\multipleapp_pages\\csv\\earth quake.csv', "Earth-Quake", "Earthquakes can strike suddenly...", "./assetes/images/Earth.jpg")
display_dataset('D:\\Space\\multipleapp_pages\\csv\\daino.csv', "Dinosaur", "Dinosaurs are a group of reptiles...", "./assetes/images/Dinosaur.jpg")
