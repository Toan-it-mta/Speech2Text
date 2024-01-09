from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainer
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import Seq2SeqTrainingArguments
import numpy as np


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def prepare_dataset(batch):
    audio = batch
    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["arrs"], sampling_rate=16000).input_features[0]
    print(batch["input_features"].shape)
    print(processor.feature_extractor.chunk_length )
    print(processor.feature_extractor.n_samples )
    # encode target text to label ids 
    batch["labels"] = processor.tokenizer(batch["gold_sentence"]).input_ids
    return batch

#Load Dataset
common_voice = DatasetDict()
common_voice = load_from_disk("/mnt/wsl/PHYSICALDRIVE0p1/toan/data_speech_to_text/aia_asr")
processor = WhisperProcessor.from_pretrained("/mnt/wsl/PHYSICALDRIVE0p1/toan/data_speech_to_text/open-ai-whisper-large-v2", language="vi", task="transcribe")
processor.feature_extractor.chunk_length = 10
processor.feature_extractor.n_samples = 10 * 16000
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=3)
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
metric = evaluate.load("wer")
model = WhisperForConditionalGeneration.from_pretrained("/mnt/wsl/PHYSICALDRIVE0p1/toan/data_speech_to_text/open-ai-whisper-large-v2")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-vi-aia",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=50000,
    max_steps=4000,
    gradient_checkpointing=True,
    # fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()