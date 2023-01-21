import yaml
from pydantic import BaseModel, dataclasses


class ModelParams(BaseModel):
    models_dir: str
    models_version: int
    token: str
    initializer_token: str
    max_prompt_length: int
    export: str


class TrainParams(BaseModel):
    epochs: int


class DatasetParams(BaseModel):
    group_datasets_dir: str
    single_datasets_dir: str
    single_prompts: list[str]
    group_prompts: list[str]
    single_urls: list[str]
    group_urls: list[str]


@dataclasses.dataclass
class Params:
    params_path: str

    def __post_init__(self):
        with open(self.params_path, "r") as f:
            parsed_params = yaml.safe_load(f)
        self.model = ModelParams(**parsed_params["model"])
        self.train = TrainParams(**parsed_params["train"])
        self.dataset = DatasetParams(**parsed_params["dataset"])
