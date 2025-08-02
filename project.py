import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Page config

st.set_page_config(page_title="Hello")

st.title("📊 Student Dropout Predictor")
#st.sidebar.image('my.jpg', width=150)
st.sidebar.header('Project Made by:')
st.sidebar.write('Abhey syal')


# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("✅ Dataset Preview")
    st.dataframe(df.head())

    # drop columns
    df.drop(columns=['roll_no', 'student_name'], inplace=True)

    # Select Target Variable

    columns = df.columns.tolist()
    default_index = columns.index('target') if 'target' in columns else 0
    target_col = st.selectbox("🎯 The Target Column (for prediction)", columns, index=default_index)

    # Encode categorical features
    label_encoders = {}
    df_encoded = df.copy()

    for col in df_encoded.select_dtypes(include='object'):
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    # Feature & Target
    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # logistic regression
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cs = classification_report(y_test, y_pred)

    st.subheader("📈 Model Evaluation")
    st.markdown(f"- Accuracy Score: {acc:.2f}")
    st.markdown(f"- Confusion Matrix:\n```\n{cm}\n```")
    st.markdown(f"- Classification Report:\n```\n{cs}\n```")

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
    st.subheader("🔮 Predict Whether the student will dropout or continue")

    # predict for new student

    new_student = {
        'sem1_result': st.number_input('Sem1_Result'),
        'sem2_result': st.number_input('Sem2_Result'),
        'total_supplies': st.number_input('Total number of supplies'),
    }

    # convert to dataFrame

    new_df = pd.DataFrame([new_student])
    st.write(new_df)

    # predict

    new_pred = model.predict(new_df)[0]
    st.write("\n Dropout Prediction (1=dropout, 0=Continue):")

    if st.button("Predict"):
        encoded_input = [
            label_encoders[feature].transform([new_df[feature]])[0]
            if feature in label_encoders else new_df[feature]
            for feature in X.columns
        ]
        encoded_input = np.array(encoded_input)
        prediction = model.predict(encoded_input.reshape(1, -1))[0]
        st.success(f"🧠 Predicted {target_col}: {prediction:.2f}")