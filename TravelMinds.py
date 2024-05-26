#!/usr/bin/env python
# coding: utf-8

# In[26]:


import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
pinecone_key = os.getenv('PINECONE_API_KEY')
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import pandas as pd
import ollama

#ollama run llama3

pc = Pinecone(api_key= pinecone_key)
index = pc.Index("myscu")
model = SentenceTransformer('all-MiniLM-L6-v2')

#These are the input the backend requires from the frontend
# change directory for 'csv_file' to where the traveldata.csv is located
csv_file = 'traveldata.csv'
#num_days = 3
#text_prompt = "park"
#city = "San Jose"

import ollama
import pandas as pd

def query_and_generate_travel_plan(text_prompt, csv_file, city, num_days):
    # Generate the embedding
    embedding = model.encode(text_prompt)
    vector_as_list = embedding.tolist()
    
    # Query the Pinecone index
    query_result = index.query(
        namespace="travel_embedding",
        vector=vector_as_list,
        top_k=6,
        include_values=False,
        include_metadata=True,
        filter={"addr_city": {"$eq": city}}
    )
    
    # Load the DataFrame
    df = pd.read_csv(csv_file, low_memory=False)
    
    # Extract IDs from the query results
    ids = [int(match['id']) for match in query_result['matches']]
    
    # Retrieve the rows corresponding to the IDs
    result_df = df.iloc[ids]
    
    descriptions = [f"{row['name']} located in {row['addr_city']}." for index, row in result_df.iterrows()]
    landmarks = "\n" + "\n".join(descriptions)
    
    # Ensure the number of days does not exceed 3
    num_days = min(num_days, 3)

    # Generate the prompt
    messages = [{
        'role': 'user',
        'content': f"""
Generate a travel itinerary with the following details:
- Location: {city}
- Landmarks: {landmarks}
- Number of days: {num_days}

For each day, provide a plan with:
- Morning: Visit one landmark
- Afternoon: Visit another landmark

Provide the itinerary in a bullet point format for each day. Use only the provided details. Start the response directly with the itinerary formatted as bullet points for each day without any introductory text or additional information.
"""
    }]
    
    # Make the request to LLaMA
    response = ollama.chat(model='llama3', messages=messages)
    travel_plan = response['message']['content']

    return travel_plan

def main():
    # Frontend UI
    image_path = "Travel_Minds_Logo.jpg"
    col1, col2 = st.columns([1, 5])
    col1.write("")
    col2.image(image_path, width = 450)
    st.markdown(
        f"<h1 style='text-align: center; font-size: 70px;font-family: URW Chancery L, cursive'>TravelMinds</h1>",
        unsafe_allow_html=True
    )
    st.sidebar.title('User Input')
    city = st.sidebar.text_input('Enter your destination (City)')
    num_days = st.sidebar.slider("Duration (days)", min_value=1, max_value=3)
    text_prompt = st.sidebar.text_area('Enter type of activities you want to do')

    if st.button('Give me Recommendations'):
        # Call backend functions with user input
        #resulting_df = query_and_retrieve_data(activities, destination, 'traveldata.csv')
        #landmarks = "\n".join([f"{row['name']} located in {row['addr_city']}." for _, row in resulting_df.iterrows()])
        travel_plan = query_and_generate_travel_plan(text_prompt, csv_file, city, num_days)

         # Display results
        st.header('Suggested Attractions Itinerary')
        st.write(travel_plan)

if __name__ == "__main__":
    main()

# In[ ]:



