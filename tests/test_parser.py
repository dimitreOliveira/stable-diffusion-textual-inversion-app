# from .context.src import parser
from src.parser import DatasetParams, ModelParams, Params, TrainParams

PARAMS_PATH = "tests/assets/params_test.yaml"
params = Params(PARAMS_PATH)


def test_Params() -> None:
    assert params.params_path == PARAMS_PATH
    assert params.model is not None
    assert params.train is not None
    assert params.dataset is not None


def test_ModelParams() -> None:
    modelParams: ModelParams = params.model
    assert modelParams.models_dir == "models/example"
    assert modelParams.token == "<token>"
    assert modelParams.initializer_token == "cat"
    assert modelParams.max_prompt_length == 77


def test_TrainParams() -> None:
    trainParams: TrainParams = params.train
    assert trainParams.epochs == 1


def test_DatasetParams() -> None:
    datasetParams: DatasetParams = params.dataset
    assert datasetParams.group_datasets_dir == "datasets/group/"
    assert datasetParams.single_datasets_dir == "datasets/single/"
    assert datasetParams.single_urls == [
        "https://i.imgur.com/VIedH1X.jpg",
        "https://i.imgur.com/eBw13hE.png",
    ]
    assert datasetParams.group_urls == [
        "https://i.imgur.com/yVmZ2Qa.jpg",
        "https://i.imgur.com/JbyFbZJ.jpg",
    ]
    assert datasetParams.single_prompts == [
        "a photo of a {}",
        "a rendering of a {}",
    ]
    assert datasetParams.group_prompts == [
        "a photo of a group of {}",
        "a rendering of a group of {}",
    ]
