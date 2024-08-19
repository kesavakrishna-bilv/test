import streamlit as st
import pandas as pd
import os
import tempfile
import json
from google.cloud import aiplatform
from google.oauth2 import service_account
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from sklearn.metrics import confusion_matrix

def convert_attrdict_to_dict(attrdict):
    """Recursively convert AttrDict to a standard Python dictionary."""
    if isinstance(attrdict, dict):
        return {key: convert_attrdict_to_dict(value) for key, value in attrdict.items()}
    elif isinstance(attrdict, list):
        return [convert_attrdict_to_dict(item) for item in attrdict]
    else:
        return attrdict

# Fetch the service account details from secrets and convert AttrDict to dict
service_account_info = st.secrets["gcp_service_account"]
service_account_dict = convert_attrdict_to_dict(service_account_info)

# Create a temporary file to store the service account key
with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
    json.dump(service_account_dict, temp_file)
    temp_file_path = temp_file.name

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the path of the temporary file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path

def predict_tabular_classification(project, endpoint_id, instance_dict, location="us-central1",
                                   api_endpoint="us-central1-aiplatform.googleapis.com"):
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    instance = json_format.ParseDict(instance_dict, Value())
    instances = [instance]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(project=project, location=location, endpoint=endpoint_id)
    response = client.predict(endpoint=endpoint, instances=instances, parameters=parameters)
    return response.predictions

st.title("AutoML Prediction using Google Cloud AI Platform")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    true_labels = df['Class'].astype(int).tolist()  # Extract true labels
    predicted_labels = []
    st.write("Uploaded CSV Data:")
    st.write(df)

    predictions_list = []
    for index, row in df.iterrows():
        instance_dict = row.to_dict()
        instance_dict = row.astype(str).to_dict()  # Convert all values to strings
        predictions = predict_tabular_classification(
            project="734821337853",
            endpoint_id="6982512363865899008",
            instance_dict=instance_dict,
            location="us-central1"
        )
        predictions_list.append(predictions)

    for i, predictions in enumerate(predictions_list):
        for prediction in predictions:
            classes = prediction['classes']
            scores = prediction['scores']
            anomaly_score = float(scores[1])  # Second score is for class '1' (anomaly)
            predicted_label = 1 if anomaly_score > 0.5 else 0
            predicted_labels.append(predicted_label)

    # Generate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    st.write("Confusion Matrix:")
    st.write(cm)

# Clean up the temporary file after use
os.remove(temp_file_path)
