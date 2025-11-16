import pandas as pd
import numpy as np

def load_df():
    df = pd.read_parquet("hf://datasets/argilla/medical-domain/data/train-00000-of-00001-67e4e7207342a623.parquet")

    def extract_label(pred):
        if isinstance(pred, (list, np.ndarray)) and len(pred) > 0 and isinstance(pred[0], dict):
            return pred[0].get("label")
        return None

    df['label'] = df['prediction'].apply(extract_label)
    df['text_length'] = df['metrics'].apply(lambda x: x.get('text_length') if isinstance(x, dict) else None)

    df = df.drop(columns=[
        'inputs', 'prediction', 'prediction_agent', 'annotation', 'annotation_agent',
        'multi_label', 'explanation', 'metadata', 'status', 'event_timestamp', 'metrics'
    ], errors='ignore')

    return df