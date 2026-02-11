import streamlit as st

st.set_page_config(page_title="HW Manager", page_icon="📚")

# Define the pages
hw1_page = st.Page("HW/HW1.py", title="HW 1", icon="📄")
hw2_page = st.Page("HW/HW2.py", title="HW 2", icon="🌐")
hw3_page = st.Page("HW/HW3.py", title="HW 3", icon="🐈")

# Create navigation
nav = st.navigation([hw1_page, hw2_page, hw3_page])

# Run the selected page
nav.run()