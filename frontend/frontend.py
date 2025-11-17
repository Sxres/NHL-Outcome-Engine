import streamlit as st 
import requests

st.set_page_config(page_title="NHL Logo Prediction", layout="centered")



col1, col2 = st.columns([16, 1], vertical_alignment="center", gap="small") 


with col1:
    st.title('NHL Logo Prediction')    
    st.subheader('Enter a Logo and get a prediction!')


input_image = st.file_uploader(label="Upload NHL Logo",accept_multiple_files=False, type=["jpg", "jpeg", "png"])

if input_image is not None:
    st.image(input_image, width=224)


#when u click a button then it makes the post request to the backend
#and gets the response
if st.button('Predict NHL Logo'): 
    if input_image is not None:
        # Prepare the file for the POST request
        files = {'file': (input_image.name, input_image.getvalue(), input_image.type)}
        
        # Make POST request to your FastAPI backend
        response = requests.post('https://backend-765461223282.northamerica-northeast2.run.app/predict', files=files)
        
        if response.status_code == 200:
            prediction = response.json()
            st.success(f"**Predicted Team:** {prediction.get('team')}")
            st.info(f"**Confidence:** {prediction.get('confidence')}")
            st.write(f"**Class ID:** {prediction.get('class_id')}")
        else:
            st.error(f"Error: {response.status_code}")
    else:
        st.warning("Please upload an image first!")



        