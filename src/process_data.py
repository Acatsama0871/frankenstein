import os
import numpy as np
from PIL import Image
from functools import partial
from datasets import Dataset, DatasetDict, load_dataset
from rich import print
from typing import Dict, Any, Iterable, Dict
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
    example["bottom_image"] = np.array(bottom_image)
    example["top_image"] = np.array(top_image)

    return example

def process_data(processor: CLIPProcessor, batched_examples: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    tokenizer = processor.tokenizer
    temp = tokenizer(batched_examples["bottom_texts"], padding=True, truncation=True, return_tensors="np")
    batched_examples["bottom_input_ids"] = temp["input_ids"]
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
    # TODO: debug
    dataset = dataset.select(range(10))
    # TODO: debug

    # load image
    load_image_func = partial(load_image, image_root_folder_path=img_root_folder_path)
    dataset = dataset.map(load_image_func, num_proc=num_proc)

    # process data
    clip_processor = CLIPProcessor.from_pretrained(
        pretrained_model_name_or_path=model_name
    )
    clip_process_func = partial(clip_processor, processor=clip_processor)
    dataset = dataset.map(clip_process_func, batched=True, num_proc=num_proc)
    # dataset.set_format(None, columns=["bottom_texts", "top_texts"])
    # print(dataset)
    # print(dataset[0].keys())
    # print(dataset[:3]["bottom_texts"])
    
    #dataset = dataset.map(clip_process_func, batched=True, num_proc=num_proc)

    # process via clip processor
    # dataset.set_format("numpy", columns=["bottom_image", "top_image"])
    # for i, example in enumerate(dataset):
    #     print(f"========{i}=======")
    #     print(example["top_image"].shape)
    #     print(example["bottom_image"].shape)
    #     # print(example["bottom_image"])
    #     # print(example["bottom_image"].shape)
    #     # print(example["top_image"].shape)
    #     print("================")

    # construct dataset
    # # load image
    # load_image_func = partial(load_image, image_root_folder_path=img_root_folder_path)
    # dataset = dataset.map(load_image_func, num_proc=num_proc)

    # # splits
    # ratios = [split_ratios["train"], split_ratios["val"], split_ratios["test"]]
    # dataset = dataset.train_test_split(train_size=ratios[0], shuffle=shuffle)
    # dataset_train = dataset["train"]
    # dataset_temp = dataset["test"]
    # dataset_temp = dataset_temp.train_test_split(test_size=ratios[2], shuffle=shuffle)
    # dataset_val = dataset_temp["train"]
    # dataset_test = dataset_temp["test"]
    # dataset = DatasetDict({"train": dataset_train, "val": dataset_val, "test": dataset_test})
