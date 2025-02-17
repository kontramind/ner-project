import os
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from berts import BertModel
from transformers import Trainer
from functools import partial
from configs import training_args, lora_config
from peft import get_peft_model
import torch


def _align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def _tokenize_and_align_labels(examples):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(_align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def _compute_metrics(eval_preds, label_names):
    print("---- Computing metrics ----")
    try:
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        metric = evaluate.load("seqeval")
        all_metrics = metric.compute(
            predictions=true_predictions, references=true_labels
        )
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {}


def main() -> None:
    model_path = "bert-base-cased"
    raw_datasets = load_dataset("conll2003", trust_remote_code=True)
    ner_feature = raw_datasets["train"].features["ner_tags"]
    label_names = ner_feature.feature.names
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    tokenized_datasets = raw_datasets.map(
        _tokenize_and_align_labels,
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

    print(f"{type(model)}")
    print(f"{model.get_nb_trainable_parameters()=}")
    compute_metrics_with_labels = partial(_compute_metrics, label_names=label_names)
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
