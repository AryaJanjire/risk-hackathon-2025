import os
import pandas as pd
from help_desk import HelpDesk
from dotenv import load_dotenv, find_dotenv
from langchain.evaluation import load_evaluator
from langchain.evaluation import EmbeddingDistance
from config import EVALUATION_DATASET

# --- Add Gork API integration ---
import requests

GORK_API_URL = "https://api.gork.ai/v1/chat"  # Example endpoint, update as needed

def gork_predict(question):
    """
    Use Gork API to get a response for the given question.
    """
    payload = {
        "prompt": question,
        "model": "gork-free",  # Change if needed
        "max_tokens": 256
    }
    response = requests.post(GORK_API_URL, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "")
    else:
        return f"Error: {response.status_code}"

def predict(model, question, use_gork=False):
    if use_gork:
        result = gork_predict(question)
        sources = None
    else:
        result, sources = model.retrieval_qa_inference(question, verbose=False)
    return result

def open_evaluation_dataset(filepath):
    df = pd.read_csv(filepath, delimiter='\t')
    return df

def get_levenshtein_distance(model, reference_text, prediction_text):
    evaluator = load_evaluator("string_distance")
    return evaluator.evaluate_strings(
        prediction=prediction_text,
        reference=reference_text
    )

def get_cosine_distance(model, reference_text, prediction_text):
    evaluator = load_evaluator("embedding_distance", distance_metric=EmbeddingDistance.COSINE)
    return evaluator.evaluate_strings(
        prediction=prediction_text,
        reference=reference_text
    )

def evaluate_dataset(model, dataset, verbose=True, use_gork=False):
    predictions = []
    levenshtein_distances = []
    cosine_distances = []
    for i, row in dataset.iterrows():
        prediction_text = predict(model, row['Questions'], use_gork=use_gork)

        # Distances
        levenshtein_distance = get_levenshtein_distance(model, row['Réponses'].strip(), prediction_text.strip())
        cosine_distance = get_cosine_distance(model, row['Réponses'].strip(), prediction_text.strip())

        if verbose:
            print("\n QUESTIONS \n", row['Questions'])
            print("\n REPONSES \n", row['Réponses'])
            print("\n PREDICTION \n", prediction_text)
            print("\n LEV DISTANCE \n", levenshtein_distance['score'])
            print("\n COS DISTANCE \n", cosine_distance['score'])

        predictions.append(prediction_text)
        levenshtein_distances.append(levenshtein_distance['score'])
        cosine_distances.append(cosine_distance['score'])

    dataset['Prédiction'] = predictions
    dataset['Levenshtein_Distance'] = levenshtein_distances
    dataset['Cosine_Distance'] = cosine_distances
    dataset.to_csv(EVALUATION_DATASET, index=False, sep= '\t')
    return dataset

def run(use_gork=False):
    dataset = open_evaluation_dataset(EVALUATION_DATASET)
    if use_gork:
        model = None
    else:
        model = HelpDesk(new_db=True)
    results = evaluate_dataset(model, dataset, use_gork=use_gork)

if __name__ == '__main__':
    load_dotenv(find_dotenv())
    use_gork = bool(int(os.environ.get("USE_GORK", "0")))  # Set USE_GORK=1 in .env to use Gork API
    if use_gork:
        model = None
    else:
        model = HelpDesk(new_db=True)
    dataset = open_evaluation_dataset(EVALUATION_DATASET)
    evaluate_dataset(model, dataset, use_gork=use_gork)

    print('Mean Levenshtein distance: ', dataset['Levenshtein_Distance'].mean())
    print('Mean Cosine distance: ', dataset['Cosine_Distance'].mean())