from typing import List, Optional

import yaml
from pydantic import BaseModel, dataclasses


class ModelParams(BaseModel):
    models_dir: str = "models"
    models_version: int = 1
    token: str = "<token>"
    initializer_token: str
    max_prompt_length: int = 77
    export: str = "True"


class TrainParams(BaseModel):
    epochs: int = 1


class DatasetParams(BaseModel):
    group_datasets_dir: str = "datasets/group/"
    single_datasets_dir: str = "datasets/single/"
    single_prompts: Optional[List[str]] = None
    group_prompts: Optional[List[str]] = None
    single_urls: Optional[List[str]] = []
    group_urls: Optional[List[str]] = []


@dataclasses.dataclass
class Params:
    params_path: str

    def __post_init__(self):
        with open(self.params_path, "r") as f:
            parsed_params = yaml.safe_load(f)
        self.model = ModelParams(**parsed_params["model"])
        self.train = TrainParams(**parsed_params["train"])
        self.dataset = DatasetParams(**parsed_params["dataset"])
