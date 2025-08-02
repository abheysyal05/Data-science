import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Page config

st.set_page_config(page_title="hello")

st.title("📊 Student Performances Predictor")


# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("✅ Dataset Preview")
    st.dataframe(df.head())

    # Select Target Variable
    target_col = st.selectbox("🎯 Select the Target Column (for prediction)", df.columns)

    # Encode categorical features
    label_encoders = {}
    df_encoded = df.copy()

    for col in df_encoded.select_dtypes(include='object'):
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    df_encoded[df_encoded.columns] = imputer.fit_transform(df_encoded)

    # Feature & Target
    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("📈 Model Evaluation")
    st.markdown(f"- Mean Squared Error: {mse:.2f}")
    st.markdown(f"- R² Score: {r2:.2f}")

    # Plot section
    st.subheader("📉 Visualization")
    col1, col2 = st.columns(2)

    with col1:
        x_axis = st.selectbox("X-axis for scatter plot", df.columns, key="scatter_x")
        y_axis = st.selectbox("Y-axis for scatter plot", df.columns, key="scatter_y")
        st.write("Scatter Plot")
        fig1, ax1 = plt.subplots()
        sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax1)
        st.pyplot(fig1)

    with col2:
        bar_col = st.selectbox("Column for bar plot", df.columns, key="bar_plot")
        st.write("Bar Plot")
        fig2, ax2 = plt.subplots()
        df[bar_col].value_counts().plot(kind='bar', ax=ax2)
        ax2.set_title(f'Distribution of {bar_col}')
        st.pyplot(fig2)

    # Prediction section
    st.subheader("🔮 Predict Math Score for New Student")

    new_data = {}
    for feature in X.columns:
        if feature in label_encoders:
            options = list(label_encoders[feature].classes_)
            new_data[feature] = st.selectbox(f"{feature}", options)
        else:
            new_data[feature] = st.number_input(f"{feature}", step=1.0)

    if st.button("Predict"):
        encoded_input = [
            label_encoders[feature].transform([new_data[feature]])[0]
            if feature in label_encoders else new_data[feature]
            for feature in X.columns
        ]
        prediction = model.predict([encoded_input])[0]
        st.success(f"🧠 Predicted {target_col}: {prediction:.2f}")


