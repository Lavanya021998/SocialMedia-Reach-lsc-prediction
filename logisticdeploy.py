# Base Libraries

import pandas as pd
import time # To delay response
import numpy as np
import warnings 
warnings.filterwarnings("ignore") # To supress warnings

import streamlit as st  # UI Module

import joblib # Saved Models Load Module
import pickle # Saved Encodings Load Module
ohe = joblib.load('ohe.pkl')
sc = joblib.load("sc.pkl")
log= joblib.load('logistic.pkl')
# Loading Saved Ordinal Encoded files
with open('Day_encoding.pkl', 'rb') as f:
    Day_encoding = pickle.load(f)
    
with open('CourseName_encoding.pkl', 'rb') as f:
    CourseName_encoding = pickle.load(f)
######### Sample Input Data to Show to the User ###############

data = pd.read_csv("inputdata.csv")
print(data.columns)
######################## Helper functions for Inputs #####################

if 'sbutton' not in st.session_state:
    st.session_state['sbutton'] = False

if 'fbutton' not in st.session_state:
    st.session_state['fbutton'] = False

def switch_sbutton_state():
    st.session_state['sbutton'] = True
    st.session_state['fbutton'] = False

def switch_fbutton_state():
    st.session_state['sbutton'] = False
    st.session_state['fbutton'] = True
###################################### Design of User Interface ################################

st.subheader(":violet[Reach of Training Institute using Social Medaia]", divider=True)
st.write(":violet[The predictive model will analyze historical data from social media platforms and estimate the expected reach of future posts.]")
colx, coly, colz = st.columns([1,2,1])

with coly:
    st.image("K.jpg")

st.divider()
st.write("Sample of Data, Predictive Model Trained...")
st.dataframe(data.head())
st.write(":green[Predicitve Model Trained on above input columns to estimate Reach of training institutes using social media.]")
st.divider()
st.subheader(":green[Predictive Modeling:]")
st.write("As our prediction value is categorical, in this Project we have trained on multi class classification algorithms,  \nLogistic Regression  \nknn algorithm  \nsvm algorithm   \nDecision Tree     \nRandom Forest  \netc..")
st.divider()
st.subheader(":red[Better Performance Model For Prediction]")
st.write("Among above we got better performance for :green[Logistic Regression Algorithm].")
st.write("Trained Logistic Regression Algorithm Used for Predictions.")
st.divider()
st.subheader(":blue[Predictions For Given Data:]")
# Prediction Buttons
cola, colb = st.columns(2)
with cola:
    fbutton = st.button("Prediction for Multiple reach columns By Uploading file...", 
                        on_click=switch_fbutton_state, icon=':material/table:', key="multi_class")
with colb:
    sbutton = st.button("Prediction for Single reach column by Entering Data.....", 
                        on_click=switch_sbutton_state, icon=':material/input:', key="single_class")
# Conditions for Predictions Based on Selection
reach_mapping = {0: 'low', 1: 'moderate', 2: 'high'}
if st.session_state['fbutton'] == True:
    file = st.file_uploader(":red[Upload Test Data File Having X Cols Shown Above:]", type=['csv','xlsx'])
    if file!=None:
        try:
            df = pd.read_csv(file)
        except:
            df = pd.read_excel(file)

        st.write(":green[Uploaded Data....]")
        st.dataframe(df.head())

        if st.button(":red[Predict]"):

            # Taking Copy of Uploaded Data
            data = df.copy()

            # Converting text columns to lower case
            for col in data.select_dtypes("O").columns:
                data[col] = data[col].str.lower()

            ############### Using above saved encoded files transforming text columns to numeric ###################
            # Binary Encoding
            data.type.replace({'post':0,'reel':1}, inplace=True)
            data.Location.replace({'hyderabad':0,'banglore':1}, inplace=True)

            # Ordinal Encoding
            data['Day'].replace(Day_encoding, inplace=True)
            data['CourseName'].replace(CourseName_encoding, inplace=True)

            # Onehot Encoding
            data_ohe= ohe.transform(data[['Institute Name']]).toarray()
            data_ohe = pd.DataFrame(data_ohe, columns=ohe.get_feature_names_out())
            data = pd.concat([data.drop(['Institute Name'], axis=1), data_ohe], axis=1)


            data[['Followers']]=sc.transform(data[['Followers']])
            expected=log.feature_names_in_
            data=data.reindex(columns=expected,fill_value=0)
            
            probabilities = log.predict_proba(data)
            predictions = log.predict(data)
            
            # Assign the highest probability category
            df['reach'] = [reach_mapping[np.argmax(prob)] for prob in probabilities]
            
            st.success("Prediction Complete!")
            st.write("Predicted Results:")
            st.dataframe(df)
            
            # Provide CSV download option
            csv = df.to_csv(index=False)
            st.download_button(label="Download Predictions as CSV",
                               data=csv,
                               file_name="reach_predictions.csv",
                               mime="text/csv")
if st.session_state['sbutton'] == True:
    st.write(":red[Enter Details of a institute:]")

    col1, col2 = st.columns(2)
    with col1:
        institute = st.selectbox("Select institute name:", data['Institute Name'].unique())
    with col2:
        coursename = st.selectbox("Select course name:", data['CourseName'].unique())

    col3, col4 = st.columns(2)
    with col3:
        followers = st.number_input(f"Enter number of followers: {data['Followers'].min()} to {data['Followers'].max()}:")
    
    with col4:
        typeofpost = st.selectbox("Select TypeofPost:", data['type'].unique())


    col5,col6 = st.columns(2)
    with col5:
        locationname=st.selectbox("select location:",data['Location'].unique())
    with col6:
        day=st.selectbox("select day:",data['Day'].unique( ))
        

    if st.button("Estimate"):

        row = pd.DataFrame([[institute,coursename,followers,typeofpost,locationname,day]], columns=data.columns)

        st.write(":green[Given social media Input Data:]")

        st.dataframe(row)

        # Feature Engineering: Need to apply same steps done for training, while giving it to model for prediction
            # Binary Encoding
        row.type.replace({'post':0,'reel':1}, inplace=True)
        row.Location.replace({'hyderabad':0,'banglore':1}, inplace=True)

        # Ordinal Encoding
        row['Day'].replace(Day_encoding, inplace=True)
        row['CourseName'].replace(CourseName_encoding,inplace=True)

        # One-Hot Encoding
        row_ohe = ohe.transform(row[['Institute Name']]).toarray()
        row_ohe = pd.DataFrame(row_ohe, columns=ohe.get_feature_names_out())
        row = pd.concat([row.drop(['Institute Name'], axis=1), row_ohe], axis=1)



        # Prediction
        print("********** Logistic Prediction ***********")
        prob0 = round(float(log.predict_proba(row)[0][0]),2)
        prob1 = round(float(log.predict_proba(row)[0][1]),2)
        prob2 = round(float(log.predict_proba(row)[0][2]),2)
        print("reach Probabilities: low - {}, moderate - {}, high - {}".format(prob0, prob1, prob2))
        print()
        out = log.predict(row)[0]
        st.write(f":blue[reach column prediction of institutes using socail media:] {out}")

st.subheader(":violet[Regression]", divider=True)

# Loading saved pickles and getting predictions


xgbr= joblib.load('xgbr.pkl')



feature_names = xgbr.feature_names_in_.tolist()

with open("feature_names.pkl",'rb') as f:
    trained_feature_names = pickle.load(f)

data1 = pd.read_csv("inputdata1.csv")

######################## Helper functions for Inputs #####################

if 'sbutton1' not in st.session_state:
    st.session_state['sbutton1'] = False


if 'fbutton1' not in st.session_state:
    st.session_state['fbutton1'] = False

def switch_sbutton_state1():
    st.session_state['sbutton1'] = True
    st.session_state['fbutton1'] = False

def switch_fbutton_state1():
    st.session_state['sbutton1'] = False
    st.session_state['fbutton1'] = True


###################################### Design of User Interface ################################

st.subheader(":orange[Predicting Likes,Shares,Comments for given Reach:]", divider=True)

st.divider()
st.write("Sample of Data, Predictive Model Trained...")
st.dataframe(data1.head())
st.write(":green[Predicitve Model Trained on above input columns to estimate Likes,Comments,Shares of training institutes posts.]")
st.divider()
st.subheader(":green[Predictive Modeling:]")
st.write("As our prediction value is Multiple Numerical , in this Project we have trained on Regression algorithms,  \nLinear Regression  \nPolynomial Regression  \nLasso&Ridge Regression  \nknn algorithm  \nsvm algorithm   \nDecision Tree     \nRandom Forest  \nXgboost  etc..")
st.divider()
st.subheader(":red[Better Performance Model For Prediction]")
st.write("Among above we got better performance for :green[Xgboost Regression Algorithm].")
st.write("Trained Xgboost Regression Algorithm Used for Predictions.")
st.divider()
st.subheader(":blue[Predictions For Given Data:]")
# Prediction Buttons

cola, colb = st.columns(2)
with cola:
    fbutton1 = st.button("Prediction of Likes,Shares,Comments of multiple rows By Uploading file...", 
                         on_click=switch_fbutton_state1, icon=':material/table:', key="multi_regress")
with colb:
    sbutton1 = st.button("Prediction of Likes,shares,Comments for Single row by Entering Data.....", 
                         on_click=switch_sbutton_state1, icon=':material/input:', key="single_regress")
# Conditions for Predictions Based on Selection
if st.session_state['fbutton1'] == True:
    file = st.file_uploader(":red[Upload Test Data File Having X Cols Shown Above:]", type=['csv','xlsx'])
    if file!=None:
        try:
            df1 = pd.read_csv(file)
        except:
            df1 = pd.read_excel(file)

        st.write(":green[Uploaded Data....]")
        st.dataframe(df1.head())

        if st.button(":red[Predict]"):

            # Taking Copy of Uploaded Data
            data1 = df1.copy()

            # Converting text columns to lower case
            for col in data1.select_dtypes("O").columns:
                data1[col] = data1[col].str.lower()

            ############### Using above saved encoded files transforming text columns to numeric ###################
            # Binary Encoding
            data1.type.replace({'post':0,'reel':1}, inplace=True)
            data1.Location.replace({'hyderabad':0,'banglore':1}, inplace=True)
            data1['Reach'].replace({'low': 0, 'moderate': 1,'high':2}, inplace=True)

            # Ordinal Encoding
            data1['Day'].replace(Day_encoding, inplace=True)
            data1['CourseName'].replace(CourseName_encoding, inplace=True)

            # Onehot Encoding
            if 'Institute Name' in data1.columns:
                data_ohe1= ohe.transform(data1[['Institute Name']]).toarray()
                data_ohe1 = pd.DataFrame(data_ohe1, columns=ohe.get_feature_names_out())

                missing_cols = set(trained_feature_names) - set(data_ohe1.columns)
                for col in missing_cols:
                    data_ohe1[col] = 0  # Add missing columns with value 0

                # Ensure column order matches training
                data_ohe1 = data_ohe1[trained_feature_names]
                # Merge with X
                data1 = pd.concat([data1.drop(['Institute Name'], axis=1), data_ohe1], axis=1)
            else:
                raise KeyError("Column 'Institute Name' not found in DataFrame X. Available columns: ", data1.columns)

            expected_features = xgbr.feature_names_in_  # Get expected feature names from trained model
            # Check for duplicate columns before reindexing
            print("Checking for duplicate columns...")
            duplicates = data1.columns[data1.columns.duplicated()]
            if len(duplicates) > 0:
                print("Duplicate columns found:", duplicates)
                # Remove duplicate columns
                data1 = data1.loc[:, ~data1.columns.duplicated()]

            # Now apply reindexing safely
            data1 = data1.reindex(columns=expected_features, fill_value=0)  
            #data1 = data1.reindex(columns=expected_features, fill_value=0)  # Ensure correct order
            data1[['Followers']] = sc.transform(data1[['Followers']])
            #expected_features=xgbr.feature_names_in_
            #data1=data1.reindex(columns=feature_names,fill_value=0)
            predictions = xgbr.predict(data1)  # Get predictions as NumPy array

            # Extract and round values for likes, shares, and comments
            df1['Predicted_Likes'] = [round(float(pred[0])) for pred in predictions]
            df1['Predicted_Shares'] = [round(float(pred[1])) for pred in predictions]
            df1['Predicted_Comments'] = [round(float(pred[2])) for pred in predictions]
            st.success(":green[Done!]")
            st.write(":blue[Predicted Likes,Comments,Shares....]")
            st.dataframe(df1)
            # Provide CSV download option
            csv = df1.to_csv(index=False)
            st.download_button(label="Download Predictions as CSV",
                                data=csv,
                                file_name="lsc_predictions.csv",
                                mime="text/csv")
if st.session_state['sbutton1'] == True:
    st.write(":red[Enter Details of post or reel:]")

    col1, col2 = st.columns(2)
    with col1:
        inst = st.selectbox("Select institute name:", data1['Institute Name'].unique(),key="institute_1")
    with col2:
        cour = st.selectbox("Select course name:", data1['CourseName'].unique(),key="course_1")

    col3, col4 = st.columns(2)
    with col3:
        foll = st.number_input(f"Enter number of followers: {data1['Followers'].min()} to {data1['Followers'].max()}:", key="followers_input")
    
    with col4:
        typep = st.selectbox("Select TypeofPost:", data1['type'].unique(),key="post_1")


    col5,col6,col7 = st.columns(3)
    with col5:
        locatio=st.selectbox("select location:",data1['Location'].unique(),key="location_1")
    with col6:
        dayname=st.selectbox("select day:",data1['Day'].unique(),key="day_1")
    with col7:
        reachp = st.selectbox("Select Reach:",data1['Reach'].unique(),key="reach_1")
    
    if st.button("Estimate",key="estimate_button_1"):

        row = pd.DataFrame([[inst,cour,foll,typep,locatio,dayname,reachp]], columns=data1.columns)

        st.write(":green[Given Input Data:]")

        st.dataframe(row)
        ############### Using above saved encoded files transforming text columns to numeric ###################
        # Binary Encoding
        row.type.replace({'post':0,'reel':1}, inplace=True)
        row.Location.replace({'hyderabad':0,'banglore':1}, inplace=True)
        row['Reach'].replace({'low': 0, 'moderate': 1,'high':2}, inplace=True)

        # Ordinal Encoding
        row['Day'].replace(Day_encoding, inplace=True)
        row['CourseName'].replace(CourseName_encoding, inplace=True)
        row = row.loc[:, ~row.columns.duplicated()]

            # Onehot Encoding
        if 'Institute Name' in row.columns:
            data_ohe1= ohe.transform(row[['Institute Name']]).toarray()
            data_ohe1 = pd.DataFrame(data_ohe1, columns=ohe.get_feature_names_out())
            data_ohe1 = data_ohe1.loc[:, ~data_ohe1.columns.duplicated()]
            row = pd.concat([row.drop(['Institute Name'], axis=1), data_ohe1], axis=1)
        else:
            raise KeyError("Column 'Institute Name' not found in DataFrame X. Available columns: ", data1.columns)
          
        row = row.loc[:, ~row.columns.duplicated()]

        print("Columns before prediction:", row.columns.tolist())
        print("Expected model features:", xgbr.feature_names_in_)
        print("Row Data:\n", row.head())

        expected_features = xgbr.feature_names_in_
        row = row.reindex(columns=expected_features, fill_value=0)

        if 'Followers' in row.columns and not row['Followers'].isna().all():
            row['Followers'] = sc.transform([[row['Followers'].values[0]]])[0][0]
        else:
            st.warning("Warning: 'Followers' value is missing or invalid.")

        # Check if row is empty before prediction
        if row.empty:
            st.error("Error: Input row is empty. Please check your input.")
        else:
            # Prediction
            predicted_values = xgbr.predict(row)[0]

            likes = round(predicted_values[0])
            shares = round(predicted_values[1])
            comments = round(predicted_values[2])

            # Display Results in Streamlit
            st.success(f"**Likes:** {likes}")
            st.success(f"**Shares:** {shares}")
            st.success(f"**Comments:** {comments}")
            


