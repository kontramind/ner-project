import os
import torch
from collections.abc import Iterator
from typing import Any, Literal, Optional, Union
from transformers import AutoModelForTokenClassification
from torch.nn.parameter import Parameter
from models.configs import lora_config
from peft import get_peft_model, PeftModel, PeftConfig


class BertModel(torch.nn.Module):
    def __init__(self, model_path, id2label, label2id):
        super(BertModel, self).__init__()
        self.model_path = model_path
        self.id2label = id2label
        self.label2id = label2id
        self.lora_config = lora_config
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_path, id2label=self.id2label, label2id=self.label2id
        )
        self.model = get_peft_model(
            self.model,
            self.lora_config,
        )
        for name, param in self.model.named_parameters():
            if "bias" in name:
                param.requires_grad = True

        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        return self.model.get_nb_trainable_parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        missing_keys, unexpected_keys = self.model.load_state_dict(
            state_dict, strict=True
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        kwargs.pop("num_items_in_batch", None)
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            task_ids=task_ids,
            **kwargs,
        )

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        selected_adapters: Optional[list[str]] = None,
        save_embedding_layers: Union[str, bool] = "auto",
        is_main_process: bool = True,
        path_initial_model_for_weight_conversion: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.model.save_pretrained(
            save_directory,
            safe_serialization,
            selected_adapters,
            save_embedding_layers,
            is_main_process,
            path_initial_model_for_weight_conversion,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        model: torch.nn.Module,
        model_id: Union[str, os.PathLike],
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        autocast_adapter_dtype: bool = True,
        ephemeral_gpu_offload: bool = False,
        low_cpu_mem_usage: bool = False,
        **kwargs: Any,
    ) -> PeftModel:
        return PeftModel.from_pretrained(
            model=model,
            model_id=model_id,
            adapter_name=adapter_name,
            is_trainable=is_trainable,
            config=config,
            autocast_adapter_dtype=autocast_adapter_dtype,
            ephemeral_gpu_offload=ephemeral_gpu_offload,
            low_cpu_mem_usage=low_cpu_mem_usage,
            kwargs=kwargs,
        )

    def save_state_dict(self, model_path):
        torch.save(self.state_dict(), model_path)

    @classmethod
    def from_state_dict(cls, model_path, id2label, label2id):
        model = BertModel("bert-base-cased", id2label, label2id)
        model.load_state_dict(torch.load(model_path))
        return model

    def save_trainable_parameters(self, model_path):
        print("---- save_trainable_parameters -----")
        trainable_params = {
            name: param.data
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        torch.save(trainable_params, model_path)

    @classmethod
    def from_trainable_parameters(cls, model_path, id2label, label2id):
        print("---- from_trainable_parameters -----")
        checkpoint = torch.load(model_path)
        model = BertModel("bert-base-cased", id2label, label2id)
        model_dict = model.state_dict()
        model_dict.update(checkpoint)  # Overwrite trainable parameters
        model.load_state_dict(model_dict)  # strict=True is default and safe here
        return model

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, Parameter]]:
        return self.model.named_parameters()
