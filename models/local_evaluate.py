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
from codecarbon import OfflineEmissionsTracker
from rich.console import Console
from rich.table import Table


def main() -> None:
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
    model = BertModel.from_trainable_parameters(file_path, id2label, label2id)

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

    tracker.start_task("train_eval")
    train_evaluation = trainer.evaluate(eval_dataset=tokenized_datasets["train"])
    train_eval_emissions = tracker.stop_task()

    tracker.start_task("validation_eval")
    validation_evaluation = trainer.evaluate(
        eval_dataset=tokenized_datasets["validation"]
    )
    validation_eval_emissions = tracker.stop_task()

    tracker.start_task("test_eval")
    test_evaluation = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    test_eval_emissions = tracker.stop_task()

    table = Table(title="Evaluation Results per Dataset")
    table.add_column("", justify="right", style="yellow", no_wrap=True)
    table.add_column("Train", justify="right", style="cyan", no_wrap=True)
    table.add_column("Validation", justify="center", style="magenta")
    table.add_column("Test", justify="left", style="green")

    table.add_row(
        "num_rows",
        f"{tokenized_datasets['train'].num_rows}",
        f"{tokenized_datasets['validation'].num_rows}",
        f"{tokenized_datasets['test'].num_rows}",
        end_section=True,
    )

    for key in train_evaluation.keys():
        table.add_row(
            f"{key}",
            f"{train_evaluation[key]}",
            f"{validation_evaluation[key]}",
            f"{test_evaluation[key]}",
        )

    table.add_section()

    table.add_row(
        "energy_consumed_kWh",
        f"{train_eval_emissions.energy_consumed}",
        f"{validation_eval_emissions.energy_consumed}",
        f"{test_eval_emissions.energy_consumed}",
    )

    table.add_row(
        "emissions_CO2eqkg",
        f"{train_eval_emissions.emissions}",
        f"{validation_eval_emissions.emissions}",
        f"{test_eval_emissions.emissions}",
    )

    console = Console()
    console.print(table)


if __name__ == "__main__":
    tracker = OfflineEmissionsTracker(
        country_iso_code="NOR",
        measure_power_secs=600,
        allow_multiple_runs=True,
        tracking_mode="process",
        project_name="bert_ner_eval_validation",
    )
    tracker.start()
    main()
    emissions = tracker.stop()
    ic(f"CO₂ Emissions: {emissions} kg CO₂")
    tracker = None
    emissions = None
