import os
import torch
import numpy as np
from PIL import Image
from functools import partial
from datasets import Dataset, DatasetDict, load_dataset
from rich import print
from typing import Dict, Any, Dict, Callable
from omegaconf import DictConfig
from transformers import CLIPProcessor


def load_image(example: Dict[str, Any], image_root_folder_path: str) -> Dict[str, Any]:
    item_id = str(example["item_id"])
    bottom_image = Image.open(
        os.path.join(image_root_folder_path, item_id, "B.jpg")
    ).convert("RGB")
    top_image = Image.open(
        os.path.join(image_root_folder_path, item_id, "U.jpg")
    ).convert("RGB")
    example["bottom_image"] = bottom_image
    example["top_image"] = top_image

    return example

def clip_process_data(batched_examples: Dict, data_processor: Callable) -> Dict:
    bottom_texts = batched_examples["bottom_text"]
    top_texts = batched_examples["top_text"]
    bottom_images = batched_examples["bottom_image"]
    top_images = batched_examples["top_image"]
    bottom_images = [np.array(i) for i in bottom_images]
    top_images = [np.array(i) for i in top_images]
    
    # process text
    bottom_text_output = data_processor.tokenizer(text=bottom_texts, padding=True, truncation=True, return_tensors="pt")
    top_text_output  = data_processor.tokenizer(text=top_texts, padding=True, truncation=True, return_tensors="pt")
    # bottom_images = data_processor.image_processor(images=np.array(bottom_images))
    # top_images = data_processor.image_processor(images=np.array(top_images))
    
    batched_examples["bottom_input_ids"] = bottom_text_output["input_ids"]
    batched_examples["bottom_attention_mask"] = bottom_text_output["attention_mask"]
    batched_examples["top_input_ids"] = top_text_output ["input_ids"]
    batched_examples["top_attention_mask"] = top_text_output ["attention_mask"]
    # batched_examples["bottom_processed_image"] = bottom_images["pixel_values"]
    # batched_examples["top_processed_image"] = top_images["pixel_values"]
    
    return batched_examples


def process_data_dataset(
    text_file_path: str, img_root_folder_path: str, save_path: str, config: DictConfig
):
    # unpack config
    num_proc = config["process_data_config"]["num_processes"]
    shuffle = config["process_data_config"]["shuffle_before_split"]
    split_ratios = config["process_data_config"]["split_ratios"]
    model_name = config["clip"]["general"]["base_model"]

    # load json file with datasets
    dataset = load_dataset("json", data_files=text_file_path, split="train")
    dataset = dataset.rename_column("Item#", "item_id")
    dataset = dataset.rename_column("bottom_texts", "bottom_text")
    dataset = dataset.rename_column("top_texts", "top_text")
    # TODO: debug
    dataset = dataset.select(range(10))
    # TODO: debug

    # load image
    load_image_func = partial(load_image, image_root_folder_path=img_root_folder_path)
    dataset = dataset.map(load_image_func)

    # process data
    clip_processor = CLIPProcessor.from_pretrained(
        pretrained_model_name_or_path=model_name
    )
    dataset = dataset.map(clip_process_data, fn_kwargs={"data_processor": clip_processor}, batched=True, num_proc=num_proc)
    print(dataset)
