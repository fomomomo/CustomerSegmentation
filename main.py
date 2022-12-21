#importing the libraries
import streamlit as st
import plotly.express as px
from PIL import Image
from skimage.transform import resize
import numpy as np
import time

# All the Functions will go here
from CustomerSegmentation import Segmentation
s = Segmentation()


# Designing the interface
st.title("Opportunity Finder for Sellers: ")
st.header("Get Customer Segments for your products")
# For newline
st.write('\n')
#Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)

st.sidebar.title("Choose a Product")
option = st.sidebar.selectbox("Which Product would you like to use for Segmentation", s.productList)
result = st.sidebar.button('Submit')




if result:
    with st.spinner('Wait for it...'):
        time.sleep(20)
    st.success('Done!')
    s.get_matched_keywords(option)
    s.get_segments()
    fig = px.scatter(s.rfm_agg, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="RFM_Level",
            hover_name="RFM_Level", size_max=100)

    fig1 = px.scatter_3d(s.df_rfm, x='Recency', y='Frequency', z='Monetary',
                    color = 'RFM_Level', opacity=0.5)
    #fig.show()
    fig1.update_traces(marker=dict(size=5), selector=dict(mode='markers'))

    tab1, tab2 = st.tabs(["Scatter Plot", "3-D Scatter Plot"])
    with tab1:
        # Use the Streamlit theme.
        # This is the default. So you can also omit the theme argument.
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with tab2:
        # Use the native Plotly theme.
        st.plotly_chart(fig1, theme="streamlit", use_container_width=True)
