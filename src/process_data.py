import os
from PIL import Image
from functools import partial
from datasets import DatasetDict, load_dataset
from typing import Dict, Any, Dict
from omegaconf import DictConfig
from transformers import CLIPProcessor


def load_image(example: Dict[str, Any], image_root_folder_path: str) -> Dict[str, Any]:
    try:
        _load_image_single(example, image_root_folder_path)
    except Exception:
        example["error"] = True
        return example

    return example


# TODO Rename this here and in `load_image`
def _load_image_single(example, image_root_folder_path):
    item_id = str(example["item_id"])
    bottom_image = Image.open(
        os.path.join(image_root_folder_path, item_id, "B.jpg")
    ).convert("RGB")
    top_image = Image.open(
        os.path.join(image_root_folder_path, item_id, "U.jpg")
    ).convert("RGB")
    example["bottom_image"] = bottom_image
    example["top_image"] = top_image
    example["error"] = False


def process_data_dataset(
    text_file_path: str, img_root_folder_path: str, save_path: str, config: DictConfig
) -> None:
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

    # load image
    load_image_func = partial(load_image, image_root_folder_path=img_root_folder_path)
    dataset = dataset.map(load_image_func, num_proc=num_proc)
    dataset = dataset.filter(lambda example: example["error"] == False)

    # process data
    clip_processor = CLIPProcessor.from_pretrained(
        pretrained_model_name_or_path=model_name
    )

    # process bottom data
    dataset = dataset.map(clip_processor.tokenizer, input_columns="bottom_text", fn_kwargs={"truncation": True, "padding": True, "return_tensors": "pt"}, batched=True, num_proc=num_proc)  # type: ignore
    dataset = dataset.rename_column("input_ids", "bottom_input_ids")
    dataset = dataset.rename_column("attention_mask", "bottom_attention_mask")
    dataset = dataset.map(clip_processor.image_processor, input_columns="bottom_image", fn_kwargs={"return_tensors": "pt"}, batched=True, num_proc=num_proc)  # type: ignore
    dataset = dataset.rename_column("pixel_values", "bottom_processed_image")
    dataset = dataset.map(
        clip_processor,
        input_columns="top_text",
        batched=True,
        fn_kwargs={"truncation": True, "padding": True, "return_tensors": "pt"},
    )
    dataset = dataset.rename_column("input_ids", "top_input_ids")
    dataset = dataset.rename_column("attention_mask", "top_attention_mask")
    dataset = dataset.map(clip_processor.image_processor, input_columns="top_image", fn_kwargs={"return_tensors": "pt"}, batched=True, num_proc=num_proc)  # type: ignore
    dataset = dataset.rename_column("pixel_values", "top_processed_image")

    # split
    dataset = dataset.shuffle(shuffle)
    dataset = dataset.flatten_indices()  # type: ignore
    dataset = dataset.train_test_split(  # type: ignore
        train_size=split_ratios["train"], shuffle=False
    )
    train_dataset = dataset["train"]
    temp_dataset = dataset["test"].train_test_split(
        test_size=split_ratios["test"] / (split_ratios["val"] + split_ratios["test"]),
        shuffle=False,
    )
    valid_dataset = temp_dataset["train"]
    test_dataset = temp_dataset["test"]

    # save
    train_dataset.save_to_disk(os.path.join(save_path, "train"))
    valid_dataset.save_to_disk(os.path.join(save_path, "valid"))
    test_dataset.save_to_disk(os.path.join(save_path, "test"))
