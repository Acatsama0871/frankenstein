import os
import gc
import tempfile
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from datasets import load_dataset, load_from_disk, concatenate_datasets
from typing import Any, Dict
from omegaconf import DictConfig
from transformers import CLIPProcessor


def load_image_func(example: Dict[str, Any], image_root_folder_path: str) -> Dict:
    try:
        item_id = str(example["item_id"])
        bottom_image = np.array(
            Image.open(os.path.join(image_root_folder_path, item_id, "B.jpg")).convert(
                "RGB"
            )
        )
        top_image = np.array(
            Image.open(os.path.join(image_root_folder_path, item_id, "U.jpg")).convert(
                "RGB"
            )
        )
    except FileNotFoundError:
        example["bottom_image"] = None
        example["top_image"] = None
        return example

    example["bottom_image"] = bottom_image
    example["top_image"] = top_image

    return example


def filtered_empty_item(example: Dict[str, Any]) -> bool:
    image_condition = (example["bottom_image"] is not None) and (
        example["top_image"] is not None
    )
    text_condition = (
        (example["top_text"] is not None)
        and (example["bottom_image"] is not None)
        and (len(example["top_text"]) > 0)
        and (len(example["bottom_text"]) > 0)
    )
    judgement_condition = (example["judgement_reason"] is not None) and (
        example["judgement"] is not None
    )
    return judgement_condition & text_condition & image_condition


def process_data_dataset(
    text_file_path: str, img_root_folder_path: str, save_path: str, config: DictConfig
) -> None:
    # unpack config
    num_proc = config["process_data_config"]["num_processes"]
    shuffle = config["process_data_config"]["shuffle_before_split"]
    split_ratios = config["process_data_config"]["split_ratios"]
    model_name = config["clip"]["general"]["base_model"]
    num_shards = config["process_data_config"]["num_shards"]
    temp_dir_base_path = config["process_data_config"]["temp_dir_base_path"]

    # load json file with datasets
    dataset = load_dataset(
        "json", data_files=text_file_path, split="train", keep_in_memory=False
    )
    dataset = dataset.rename_column("Item#", "item_id")
    dataset = dataset.rename_column("bottom_texts", "bottom_text")
    dataset = dataset.rename_column("top_texts", "top_text")

    # get feature processors
    clip_processor = CLIPProcessor.from_pretrained(
        pretrained_model_name_or_path=model_name
    )

    # process each shard
    shard_paths = []
    with tempfile.TemporaryDirectory(dir=temp_dir_base_path) as temp_dir:
        # split the dataset to shards to avoid OOM
        for i in tqdm(range(num_shards)):
            cur_shard = dataset.shard(num_shards=num_shards, index=i)

            # load image
            cur_shard = cur_shard.map(
                lambda examples: load_image_func(examples, img_root_folder_path),
                num_proc=num_proc,
            )
            cur_shard = cur_shard.filter(filtered_empty_item, num_proc=num_proc)

            # process image
            cur_shard.set_format(type="numpy")
            cur_shard = cur_shard.map(
                lambda x: clip_processor.image_processor(x),
                input_columns="bottom_image",
                num_proc=num_proc,
            )  # type: ignore
            cur_shard = cur_shard.rename_column(
                "pixel_values", "bottom_processed_image"
            )
            cur_shard = cur_shard.map(
                lambda x: clip_processor.image_processor(x),
                input_columns="top_image",
                num_proc=num_proc,
            )  # type: ignore
            cur_shard = cur_shard.rename_column("pixel_values", "top_processed_image")
            cur_shard.reset_format()

            # process text
            cur_shard = cur_shard.map(
                clip_processor.tokenizer,
                input_columns="bottom_text",
                fn_kwargs={"truncation": True, "padding": True, "return_tensors": "pt"},
                batched=True,
                num_proc=num_proc,
            )  # type: ignore
            cur_shard = cur_shard.rename_column("input_ids", "bottom_input_ids")
            cur_shard = cur_shard.rename_column(
                "attention_mask", "bottom_attention_mask"
            )
            cur_shard = cur_shard.map(
                clip_processor,
                input_columns="top_text",
                batched=True,
                fn_kwargs={"truncation": True, "padding": True, "return_tensors": "pt"},
            )
            cur_shard = cur_shard.rename_column("input_ids", "top_input_ids")
            cur_shard = cur_shard.rename_column("attention_mask", "top_attention_mask")
            cur_shard.save_to_disk(os.path.join(temp_dir, f"{i}"))
            shard_paths.append(os.path.join(temp_dir, f"{i}"))

            # release memory
            del cur_shard
            gc.collect()

        # concat shards
        shards = [load_from_disk(cur_path) for cur_path in shard_paths]
        dataset = concatenate_datasets(shards)

        # split
        dataset = dataset.shuffle(shuffle)
        dataset = dataset.flatten_indices()  # type: ignore
        dataset = dataset.train_test_split(  # type: ignore
            train_size=split_ratios["train"], shuffle=False
        )
        train_dataset = dataset["train"]
        temp_dataset = dataset["test"].train_test_split(
            test_size=split_ratios["test"]
            / (split_ratios["val"] + split_ratios["test"]),
            shuffle=False,
        )
        valid_dataset = temp_dataset["train"]
        test_dataset = temp_dataset["test"]

        # save
        train_dataset.save_to_disk(os.path.join(save_path, "train"))
        valid_dataset.save_to_disk(os.path.join(save_path, "valid"))
        test_dataset.save_to_disk(os.path.join(save_path, "test"))
