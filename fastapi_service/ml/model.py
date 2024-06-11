from dataclasses import dataclass
from pathlib import Path
import yaml
from transformers import pipeline
from typing import Dict, Any

from train.train import Autoencoder_Model
from prepData.prepData import PrepData

TASK = "sentiment-analysis"
MODEL = "cointegrated/rubert-tiny-sentiment-balanced"


# # load config file
# config_path = Path(__file__).parent / "config.yaml"
# with open(config_path, "r") as file:
#     config = yaml.load(file, Loader=yaml.FullLoader)


# @dataclass
# class SentimentPrediction:
#     """Class representing a sentiment prediction result."""

#     label: str
#     score: float

@dataclass
class JsonInfo:
    """Class representing a sentiment prediction result."""

    mse: float
    rmse: float


# def load_model():
#     """Load a pre-trained sentiment analysis model.

#     Returns:
#         model (function): A function that takes a text input and returns a SentimentPrediction object.
#     """
#     model_hf = pipeline(TASK, model=MODEL, device=-1)

#     def model(text: str) -> SentimentPrediction:
#         pred = model_hf(text)
#         pred_best_class = pred[0]
#         return SentimentPrediction(
#             label=pred_best_class["label"],
#             score=pred_best_class["score"],
#         )

#     return model

def load_model():
    """Load a pre-trained sentiment analysis model.

    Returns:
        model (function): A function that takes a text input and returns a SentimentPrediction object.
    """

    model_class = Autoencoder_Model()
    model_hf = model_class.load_model_from_MlFlow()

    def model(data_dict: Dict[str, Any]) -> JsonInfo:

        prep_class = PrepData()

        numpy_from_data = prep_class.json_to_numpy()

        mse_rmse_dict: Dict[str, Any] = model_class.start_active_validate(numpy_from_data)

        # keys = list(data_dict.keys())
        # first_value = data_dict[keys[0]]
        # last_value = data_dict[keys[-1]]
        return JsonInfo(
            mse=mse_rmse_dict['MSE'],
            rmse=mse_rmse_dict['RMSE'],
        )

    return model
