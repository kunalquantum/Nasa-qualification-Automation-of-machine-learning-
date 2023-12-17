import streamlit as st
import plotly.express as px
import pandas as pd
import datetime

# Set the title and description
st.set_page_config(page_icon=":linked_paperclips:", layout="wide")
st.title(" :computer: Data Dashboard")
st.write("An interactive dashboard with various features and best practices.")

# Upload CSV file (optional)
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Data Preprocessing: Convert non-numeric columns to numeric if possible
    for column in data.columns:
        if data[column].dtype == 'object':
            try:
                data[column] = pd.to_numeric(data[column])
            except ValueError:
                pass  # Ignore columns that can't be converted to numeric

    st.header("Data Preprocessing")
    st.write("Preview of Uploaded Data:")
    st.write(data.head())

    st.sidebar.header("Customize Your Dashboard")

    # Sidebar for user input
    selected_metrics = st.sidebar.multiselect("Select Attributes to Display", data.columns)

    # Date column selection
    date_column = st.sidebar.selectbox("Select Date Column", data.columns)

    # Ask the user whether they are naive or expert
    expertise_level = st.sidebar.radio(
        "Choose Your Expertise Level",
        ("Naive User", "Expert User")
    )

    st.header("Data Visualization")

    if expertise_level == "Naive User":
        st.write("You have selected 'Naive User' mode. Let's start by visualizing your data step by step.")
        
        # Line Chart
        st.subheader("Line Chart")
        st.write("Use: Line charts are ideal for visualizing trends and changes over time.")
        st.write("Description: Line charts consist of data points connected by lines, making it easy to see how selected attributes change over time.")
        
        if len(selected_metrics) >= 2:
            st.write(f"Displaying selected attributes: {', '.join(selected_metrics)}")
            line_chart = px.line(data, x=data.index, y=selected_metrics, title="Attributes Over Time")
            st.plotly_chart(line_chart)
            st.write("This line chart shows how the selected attributes change over time. It helps visualize trends and patterns over a continuous time axis.")
        else:
            st.warning("Please select at least two attributes for the Line Chart.")

        # Bar Chart
        st.subheader("Bar Chart")
        st.write("Use: Bar charts are suitable for comparing categorical data or discrete values.")
        st.write("Description: Bar charts use bars to represent data, making it easy to compare values across different categories.")
        
        if len(selected_metrics) >= 2:
            st.write(f"Displaying selected attributes: {', '.join(selected_metrics)}")
            bar_chart = px.bar(data, x=data.index, y=selected_metrics, title="Attributes Over Time")
            st.plotly_chart(bar_chart)
            st.write("The bar chart displays the selected attributes as bars over time. It is useful for comparing values across different time points.")
        else:
            st.warning("Please select at least two attributes for the Bar Chart.")

        # Scatter Plot
        st.subheader("Scatter Plot")
        st.write("Use: Scatter plots are used to visualize relationships between two numeric variables.")
        st.write("Description: Scatter plots display data points on a graph, making it easy to identify patterns and correlations.")
        
        if len(selected_metrics) >= 2:
            st.write(f"Displaying selected attributes: {', '.join(selected_metrics)}")
            scatter_plot = px.scatter(data, x=selected_metrics[0], y=selected_metrics[1], title="Scatter Plot")
            st.plotly_chart(scatter_plot)
            st.write("This scatter plot visualizes the relationship between two selected attributes, making it easy to identify patterns and correlations.")
        else:
            st.warning("Please select at least two attributes for the Scatter Plot.")

        # Box Plot
        st.subheader("Box Plot")
        st.write("Use: Box plots show the distribution, variability, and potential outliers of a dataset.")
        st.write("Description: Box plots provide a summary of the distribution of data and help identify outliers and variability.")
        
        if len(selected_metrics) >= 1:
            st.write(f"Displaying selected attributes: {', '.join(selected_metrics)}")
            box_plot = px.box(data, x=data.index, y=selected_metrics, title="Box Plot")
            st.plotly_chart(box_plot)
            st.write("The box plot displays the distribution, variability, and potential outliers of the selected attributes.")
        else:
            st.warning("Please select at least one attribute for the Box Plot.")

        # Heatmap
        st.subheader("Heatmap")
        st.write("Use: Heatmaps are used to visualize correlations or relationships between variables.")
        st.write("Description: Heatmaps use colors to represent the magnitude or intensity of values. They are effective for identifying patterns and relationships between two or more attributes.")
        
        if len(selected_metrics) >= 2:
            st.write(f"Displaying selected attributes: {', '.join(selected_metrics)}")
            heatmap_data = data[selected_metrics]
            heatmap = px.imshow(heatmap_data.corr(), x=heatmap_data.columns, y=heatmap_data.columns, title="Correlation Heatmap")
            st.plotly_chart(heatmap)
            st.write("The correlation heatmap above shows how selected attributes correlate with each other. Darker colors indicate stronger correlations.")
        else:
            st.warning("Please select at least two attributes for the Heatmap.")

        # # Area Chart
        # st.subheader("Area Chart")
        # st.write("Use: Area charts are ideal for visualizing cumulative data over time.")
        # st.write("Description: Area charts show cumulative data trends over time, making it easy to compare the total values between different categories.")
        
        # if len(selected_metrics) >= 2:
        #     st.write(f"Displaying selected attributes: {', '.join(selected_metrics)}")
        #     data[date_column] = pd.to_datetime(data[date_column])
        #     data = data.set_index(date_column)
        #     area_chart = px.area(data, x=data.index, y=selected_metrics, title="Cumulative Attributes Over Time")
        #     st.plotly_chart(area_chart)
        #     st.write("The area chart above displays how the selected attributes accumulate over time.")
        # else:
        #     st.warning("Please select at least two attributes for the Area Chart.")

            
            
    elif expertise_level == "Expert User":
        st.write("You have selected 'Expert User' mode. You can customize the dashboard based on your preferences.")
        
        # Choose the type of visualization
        visualization_type = st.sidebar.radio(
            "Choose Visualization Type",
            ("Line Chart", "Bar Chart", "Scatter Plot", "Box Plot", "Heatmap", "Area Chart")
        )
        
        if not uploaded_file:
            st.warning("Please upload a CSV file to get started.")
        else:
            if not selected_metrics:
                st.warning("Please select at least one attribute to display.")
            else:
                if st.button("Generate Visualization"):
                    st.subheader(f"{visualization_type} Visualization")
                    
                    if visualization_type == "Line Chart":
                        data[date_column] = pd.to_datetime(data[date_column])
                        data = data.set_index(date_column)
                        st.write(f"Displaying selected attributes: {', '.join(selected_metrics)}")
                        line_chart = px.line(data, x=data.index, y=selected_metrics, title="Attributes Over Time")
                        st.plotly_chart(line_chart)
                        st.write("This line chart shows how the selected attributes change over time. It helps visualize trends and patterns over a continuous time axis.")
                    
                    elif visualization_type == "Bar Chart":
                        st.write(f"Displaying selected attributes: {', '.join(selected_metrics)}")
                        bar_chart = px.bar(data, x=data.index, y=selected_metrics, title="Attributes Over Time")
                        st.plotly_chart(bar_chart)
                        st.write("The bar chart displays the selected attributes as bars over time. It is useful for comparing values across different time points.")
                    
                    elif visualization_type == "Scatter Plot":
                        st.write(f"Displaying selected attributes: {', '.join(selected_metrics)}")
                        scatter_plot = px.scatter(data, x=selected_metrics[0], y=selected_metrics[1], title="Scatter Plot")
                        st.plotly_chart(scatter_plot)
                        st.write("This scatter plot visualizes the relationship between two selected attributes, making it easy to identify patterns and correlations.")
                    
                    # Additional visualizations for Expert Users
                    elif visualization_type == "Box Plot":
                        st.write(f"Displaying selected attributes: {', '.join(selected_metrics)}")
                        box_plot = px.box(data, x=selected_metrics[0], y=selected_metrics[1], title="Box Plot")
                        st.plotly_chart(box_plot)
                        st.write("The box plot displays the distribution, variability, and potential outliers of the selected attributes.")
                    
                    elif visualization_type == "Heatmap":
                        st.write(f"Displaying selected attributes: {', '.join(selected_metrics)}")
                        heatmap_data = data[selected_metrics]
                        st.write("Use: Heatmaps are used to visualize correlations or relationships between variables.")
                        st.write("Description: Heatmaps use colors to represent the magnitude or intensity of values. They are effective for identifying patterns and relationships between two or more attributes.")
                        
                        heatmap = px.imshow(heatmap_data.corr(), x=heatmap_data.columns, y=heatmap_data.columns,
                                            title="Correlation Heatmap")
                        st.plotly_chart(heatmap)
                        st.write("The correlation heatmap above shows how selected attributes correlate with each other. Darker colors indicate stronger correlations.")
                    
                    elif visualization_type == "Area Chart":
                        st.write(f"Displaying selected attributes: {', '.join(selected_metrics)}")
                        data[date_column] = pd.to_datetime(data[date_column])
                        data = data.set_index(date_column)
                        st.write("Use: Area charts are ideal for visualizing cumulative data over time.")
                        st.write("Description: Area charts show cumulative data trends over time, making it easy to compare the total values between different categories.")
                        
                        area_chart = px.area(data, x=data.index, y=selected_metrics, title="Cumulative Attributes Over Time")
                        st.plotly_chart(area_chart)
                        st.write("The area chart above displays how the selected attributes accumulate over time.")
    
    # Export Data
    if st.sidebar.button("Export Data as CSV"):
        st.sidebar.download_button("Click to Download Data", data.to_csv(index=False), key='download_data')

    # Dashboard Footer and Contact Information
    st.sidebar.markdown("For questions or support, contact us at [example@email.com](mailto:example@email.com).")
    st.sidebar.markdown("© 2023 Your Company")

    st.sidebar.header("Sample Data")
    st.sidebar.write("You can also use the sample data below.")
    st.sidebar.write("Sample Data:")
    st.sidebar.write(data.describe())

st.sidebar.write("No data uploaded yet.")