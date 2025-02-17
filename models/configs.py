from peft import LoraConfig, TaskType
from transformers import TrainingArguments


lora_config = LoraConfig(
    r=8,
    task_type=TaskType.TOKEN_CLS,
    inference_mode=False,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="all",
)


steps = 100
epochs = 1
batch_size = 8


training_args = TrainingArguments(
    output_dir="data/models",
    num_train_epochs=epochs,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=steps,
    disable_tqdm=False,
    logging_dir="data/logs",
    logging_steps=steps,
    push_to_hub=False,
    remove_unused_columns=False,
    do_eval=True,
)
