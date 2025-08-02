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

