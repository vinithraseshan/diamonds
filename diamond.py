import streamlit as st
import pandas as pd
import pickle
import streamlit as st
import pandas as pd
import pickle


diamond = pd.read_csv('diamonds.csv')


with open("training_columns.pkl", "rb") as f:
    train_columns = pickle.load(f)

model_pickle = open('rfdiamond.pickle', 'rb') 
reg_model = pickle.load(model_pickle) 
model_pickle.close()

mapie_pickle = open('mapie.pickle', 'rb') 
mapie_model = pickle.load(mapie_pickle) 
mapie_pickle.close()

st.header("Diamonds Prices Predictor 💎 ")
st.write('This app helps you estimate diamond prices based on selected features.')
st.image('diamond_image.jpg')

alpha = st.slider("Select alpha value for prediction intervals", min_value=0.01, max_value=0.50, value=0.1)
confidence = 1 - alpha
st.header('Predicting Prices...')


st.sidebar.image('diamond_sidebar.jpg')
st.sidebar.markdown("**Diamond Features Input**")
st.sidebar.write("You can either upload your data file or manually enter diamond features")


with st.sidebar.expander("Option 1: Upload CSV file"):
    st.text("Upload a CSV file containing the diamond details")
    userinput = st.file_uploader("Choose a CSV file", type='csv')
    st.markdown("**Sample Data Format for Upload**")
    st.dataframe(diamond.head(5).drop(columns = ['price']))
    st.write("Ensure your uploaded file has the same column names and data types as shown above")


with st.sidebar.expander("Option 2: Fill out Form"):
    with st.form("user_input"):
        carat = st.number_input('Carat Weight', min_value=diamond['carat'].min(), max_value=diamond['carat'].max(), step=0.01)
        cut = st.selectbox(label='Cut Quality', options=['Fair', 'Good', 'Very Good', 'Ideal', 'Premium'])
        color = st.selectbox(label='Diamond Color', options=['J', 'I', 'H', 'G', 'F', 'E', 'D'])
        clarity = st.selectbox('Clarity', options=['I1', 'SI2', 'SI1', 'VS1', 'VS2', 'VVS2', 'VVS1', 'IF'])
        depthpercent = st.number_input('Depth(%)', min_value=diamond['depth'].min(), max_value=diamond['depth'].max(), step=0.1)
        table = st.number_input('Table(%)', min_value=diamond['table'].min(), max_value=diamond['table'].max(), step=0.01)
        length = st.number_input('Length(mm)', min_value=diamond['x'].min(), max_value=diamond['x'].max(), step=0.01)
        width = st.number_input('Width(mm)', min_value=diamond['y'].min(), max_value=diamond['y'].max(), step=0.01)
        depth = st.number_input('Depth(mm)', min_value=diamond['z'].min(), max_value=diamond['z'].max(), step=0.01)
        button = st.form_submit_button('Submit Form Data')


if userinput is not None:  
    st.write("CSV file successfully uploaded")
    st.write(f"Predicted Price with {confidence*100} % Confidence")
    user_data = pd.read_csv(userinput)
    features = user_data.copy()
    features = pd.get_dummies(features, drop_first=True) 
    #used chat gpt to debug columns not matching training columns
    missing_cols = set(train_columns) - set(features.columns)
    for col in missing_cols:
        features[col] = 0  
    features = features[train_columns]
    y_pred, intervals = mapie_model.predict(features, alpha=alpha)
    user_data['predicted price'] = y_pred
    user_data['lower price limit'] = [max(0, interval[0]) for interval in intervals]  
    user_data['upper price limit'] = [interval[1] for interval in intervals] 
    st.write(user_data)

elif button:  
    st.write("Form submitted successfully")
    encode_df = pd.DataFrame({
        'carat': [carat],
        'cut': [cut],
        'color': [color],
        'clarity': [clarity],
        'depth': [depthpercent],
        'table': [table],
        'x': [length],
        'y': [width],
        'z': [depth]
    })

    encode_df = pd.get_dummies(encode_df, columns=['cut', 'color', 'clarity'], drop_first=True)
    missing_cols = set(train_columns) - set(encode_df.columns)
    for col in missing_cols:
        encode_df[col] = 0 
    encode_df = encode_df[train_columns]
    prediction, intervals = mapie_model.predict(encode_df, alpha=alpha)
    pred_value = prediction[0] 
    lower_limit = max(0,intervals[0][0])  
    upper_limit = intervals[0][1]  
    st.metric(label="Predicted Price", value=f"${pred_value:.2f}")
    st.write(f"{confidence*100}% **Confidence Interval**: [${lower_limit}, ${upper_limit}]")
    
else:
    st.write("Please upload a file or fill out the form to make a prediction.")


st.write("### Model Insights")

tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", 
                            "Histogram of Residuals", 
                            "Predicted Vs. Actual", 
                            "Coverage Plot"])
with tab1:
    st.write("### Feature Importance")
    st.image('feature_imp.svg')
    st.caption("Relative importance of features in prediction.")
with tab2:
    st.write("### Histogram of Residuals")
    st.image('residual_plot.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")
with tab3:
    st.write("### Plot of Predicted Vs. Actual")
    st.image('pred_vs_actual.svg')
    st.caption("Visual comparison of predicted and actual values.")
with tab4:
    st.write("### Coverage Plot")
    st.image('coverage.svg')
    st.caption("Range of predictions with confidence intervals.")
