import torch
from icecream import ic
from datasets import load_dataset
from transformers import AutoTokenizer
from berts import BertModel
from codecarbon import OfflineEmissionsTracker


def _extract_entities(predictions, offset_mapping, id2label):
    """
    Extract entities from model predictions using the offset mapping.

    Args:
        predictions (Tensor): Tensor of predicted class indices for each token.
        offset_mapping (Tensor): Tensor of (start, end) offsets for each token.
        id2label (dict): Mapping from class index to label string.

    Returns:
        list of tuples: Each tuple is (start, end, label) for an extracted entity.
    """
    entities = []
    idx = 0
    while idx < len(predictions):
        # Skip special tokens with offset (0,0)
        start_off, end_off = offset_mapping[idx]
        if start_off == 0 and end_off == 0:
            idx += 1
            continue

        label = id2label[predictions[idx].item()]
        if label != "O":
            first_label = label
            entity_type = label.split("-")[-1]  # e.g., "PER", "ORG", etc.
            start = start_off.item() if hasattr(start_off, "item") else start_off

            # Group consecutive tokens that belong to the same entity.
            j = idx + 1
            while j < len(predictions):
                next_start, next_end = offset_mapping[j]
                if next_start == 0 and next_end == 0:
                    j += 1
                    continue
                next_label = id2label[predictions[j].item()]
                # Group tokens if they are continuation tokens (I-*) of the same entity.
                if (
                    next_label != "O"
                    and next_label.endswith(entity_type)
                    and next_label.startswith("I")
                ):
                    j += 1
                else:
                    break
            last_offset = offset_mapping[j - 1][1]
            end = last_offset.item() if hasattr(last_offset, "item") else last_offset
            entities.append((start, end, first_label))
            idx = j  # Move index past this entity group.
        else:
            idx += 1
    return entities


def _rebuild_annotated_text(text, entities):
    """
    Rebuild the annotated text by inserting entity annotations after each grouped entity.

    Args:
        text (str): The original text.
        entities (list): List of tuples (start, end, label) for each entity.

    Returns:
        str: The annotated text.
    """
    last_end = 0
    annotated_text = ""
    for start, end, label in sorted(entities, key=lambda x: x[0]):
        annotated_text += text[last_end:start]  # Text before the entity.
        annotated_text += (
            text[start:end] + f"[#{label}]"
        )  # Append entity span with its annotation.
        last_end = end
    annotated_text += text[last_end:]
    return annotated_text


def main() -> None:
    model_path = "bert-base-cased"
    raw_datasets = load_dataset("conll2003", trust_remote_code=True)
    ner_feature = raw_datasets["train"].features["ner_tags"]

    label_names = ner_feature.feature.names
    labels = raw_datasets["train"][0]["ner_tags"]
    labels = [label_names[i] for i in labels]
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

    text = "My name is Sasha and I work at DNV in Høvik. Høvik is in Norway. Norway is not a member of European Union."

    file_path = "data/models/test-trainable-paramaters.pt"
    model = BertModel.from_trainable_parameters(file_path, id2label, label2id)

    # Tokenize the input text with offset mappings
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    inputs = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
    # Remove offset_mapping from inputs so it isn't passed to the model.
    offset_mapping = inputs.pop("offset_mapping")[0]  # shape: (num_tokens, 2)

    # get predictions.
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)[0]  # shape: (num_tokens,)

    entities = _extract_entities(predictions, offset_mapping, id2label)
    annotated_text = _rebuild_annotated_text(text, entities)

    ic(text)
    ic(annotated_text)


if __name__ == "__main__":
    tracker = OfflineEmissionsTracker(
        country_iso_code="NOR",
        measure_power_secs=600,
        allow_multiple_runs=True,
        tracking_mode="process",
        project_name="bert_ner_inference",
    )
    tracker.start()
    main()
    emissions = tracker.stop()
    ic(f"CO₂ Emissions: {emissions} kg CO₂")
    tracker = None
    emissions = None
