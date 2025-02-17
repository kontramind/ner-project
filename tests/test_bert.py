import torch
from models.berts import BertModel


def test_instantiate_model() -> None:
    model = BertModel(model_path="bert-base-cased", id2label=[], label2id=[])
    assert model is not None


def test_state_dict_contains_bias():
    """Test that state_dict contains bias"""

    model = BertModel(model_path="bert-base-cased", id2label=[], label2id=[])
    state_dict = model.state_dict()

    # Check each key in returned state dict
    assert any("bias" in key for key in state_dict), (
        "State dict does not contain bias parameters at all"
    )


def test_state_dict_contains_lora():
    """Test that state_dict contains LoRA"""

    model = BertModel(model_path="bert-base-cased", id2label=[], label2id=[])
    state_dict = model.state_dict()

    # Check each key in returned state dict
    assert any("lora" in key for key in state_dict), (
        "State dict does not contain LoRA parameters at all"
    )


def test_nb_trainabøe_parameters_lt_all_paramateres():
    model = BertModel(model_path="bert-base-cased", id2label=[], label2id=[])
    nb_trainable_parameters, nb_all_paramaters = model.get_nb_trainable_parameters()
    assert nb_trainable_parameters < nb_all_paramaters, (
        "Number of trainable parameters should be less than the number of all parameters"
    )


def test_state_dict_parameter_types_are_tensor():
    """Test that parameters are of correct type"""
    model = BertModel(model_path="bert-base-cased", id2label=[], label2id=[])
    state_dict = model.state_dict()

    # All parameters should be torch tensors
    assert all(isinstance(param, torch.Tensor) for param in state_dict.values()), (
        "All parameters should be torch tensors"
    )


def test_save_and_load_trainable_parameters():
    file_path = "unittest_save_and_load_trainable_parameters.pt"
    model = BertModel(model_path="bert-base-cased", id2label=[], label2id=[])
    model.save_trainable_parameters(file_path)

    model = None
    assert model is None

    model = BertModel.from_trainable_parameters(file_path, id2label=[], label2id=[])
    assert model is not None


def test_save_and_from_trainable_parameters():
    file_path = "data/models/unittest_save_and_load_trainable_parameters.pt"

    model = BertModel(model_path="bert-base-cased", id2label=[], label2id=[])
    model.save_trainable_parameters(file_path)

    model_loaded = BertModel.from_trainable_parameters(
        file_path, id2label=[], label2id=[]
    )
    params = dict(model.named_parameters())
    params_loaded = dict(model_loaded.named_parameters())

    assert params.keys() == params_loaded.keys(), (
        "Models have different parameter names"
    )

    for name, param in params.items():
        param_loaded = params_loaded[name]

        # Check if shapes match
        assert param.shape == param_loaded.shape, (
            f"Shape mismatch for {name}: {param.shape} vs {param_loaded.shape}"
        )

        # Check if values are exactly equal
        assert torch.equal(param, param_loaded), (
            f"Value mismatch for {name}: {param.shape} vs {param_loaded.shape}"
        )
