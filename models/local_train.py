import os
from icecream import ic
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from berts import BertModel
from transformers import Trainer
from functools import partial
from configs import training_args
from utils import tokenize_and_align_labels, compute_metrics


def main() -> None:
    ic()
    model_path = "bert-base-cased"
    raw_datasets = load_dataset("conll2003", trust_remote_code=True)
    ner_feature = raw_datasets["train"].features["ner_tags"]
    label_names = ner_feature.feature.names
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    tokenized_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    labels = raw_datasets["train"][0]["ner_tags"]
    labels = [label_names[i] for i in labels]
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

    file_path = "data/models/test-trainable-paramaters.pt"

    if os.path.isfile(file_path):
        model = BertModel.from_trainable_parameters(file_path, id2label, label2id)
    else:
        model = BertModel(
            model_path=model_path,
            id2label=id2label,
            label2id=label2id,
        )

    ic(model.get_nb_trainable_parameters())
    compute_metrics_with_labels = partial(compute_metrics, label_names=label_names)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_with_labels,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.evaluate()
    model.save_trainable_parameters(file_path)


if __name__ == "__main__":
    main()
