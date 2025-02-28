import evaluate
import numpy as np
from icecream import ic
from transformers import AutoTokenizer


def align_labels_with_tokens(labels, word_ids):
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


def tokenize_and_align_labels(examples):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def compute_metrics(eval_preds, label_names):
    ic()
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
            "micro_precision": all_metrics["overall_precision"],
            "micro_recall": all_metrics["overall_recall"],
            "micro_f1": all_metrics["overall_f1"],
            "micro_accuracy": all_metrics["overall_accuracy"],
            "entity_PER_precision": all_metrics["PER"]["precision"],
            "entity_PER_recall": all_metrics["PER"]["recall"],
            "entity_PER_f1": all_metrics["PER"]["f1"],
            "entity_ORG_precision": all_metrics["ORG"]["precision"],
            "entity_ORG_recall": all_metrics["ORG"]["recall"],
            "entity_ORG_f1": all_metrics["ORG"]["f1"],
            "entity_LOC_precision": all_metrics["LOC"]["precision"],
            "entity_LOC_recall": all_metrics["LOC"]["recall"],
            "entity_LOC_f1": all_metrics["LOC"]["f1"],
            "entity_MISC_precision": all_metrics["MISC"]["precision"],
            "entity_MISC_recall": all_metrics["MISC"]["recall"],
            "entity_MISC_f1": all_metrics["MISC"]["f1"],
        }

    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {}
