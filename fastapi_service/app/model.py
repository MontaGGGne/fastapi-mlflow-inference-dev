from dataclasses import dataclass
from pathlib import Path
import yaml
import json
import traceback
from transformers import pipeline
from typing import List, Dict, Any

from train.train import Autoencoder_Model
from prepData.prepData import PrepData

URI = "https://dagshub.com/Dimitriy200/"
NAME_MODEL = "autoencoder2"
VERSION_MODEL = "latest"


@dataclass
class AutoencoderModelPrediction:
    """Class representing a sentiment prediction result."""

    mse: float | str = "None value"
    rmse: float | str = "None value"
    error: str = "No Error"


def load_autoencoder_model():
    """Load a pre-trained sentiment analysis model.

    Returns:
        model (function): A function that takes a text input and returns a SentimentPrediction object.
    """

    model_class = Autoencoder_Model()
    model_hf = model_class.load_model_from_MlFlow(token="ggg")

    def model(data_dict: List[Dict[str, Any]]) -> AutoencoderModelPrediction:

        prep_class = PrepData()

        try:
            numpy_from_data = prep_class.json_to_numpy(data_dict)
            predict_data = model_class.start_predict_model(model_hf, numpy_from_data)
        except Exception:
            return AutoencoderModelPrediction(
                error=f"ERROR: data_dict ({data_dict}) caused an error - {traceback.format_exc()}"
            )
        

        mse_rmse_res: Dict[str, Any] = model_class.start_active_validate(predict_data, numpy_from_data)

        return AutoencoderModelPrediction(
            mse=mse_rmse_res['MSE'],
            rmse=mse_rmse_res['RMSE'],
        )

    return model
