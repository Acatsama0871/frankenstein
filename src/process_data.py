import os
import numpy as np
from PIL import Image
from datasets import load_dataset
from typing import Dict, Any, Dict, Tuple
from omegaconf import DictConfig
from rich import print
from transformers import CLIPProcessor
from tqdm.auto import tqdm
from functools import partial
from concurrent.futures import ThreadPoolExecutor


def load_image_func(example: Dict[str, Any], image_root_folder_path: str) -> Tuple:
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
    except Exception:
        return None, None, False

    return bottom_image.tolist(), top_image.tolist(), True


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
    load_image_func_with_root = partial(
        load_image_func, image_root_folder_path=img_root_folder_path
    )
    with ThreadPoolExecutor(max_workers=num_proc) as executor:
        result = list(
            tqdm(executor.map(load_image_func_with_root, dataset), total=len(dataset))
        )
    bottom_image_list = []
    top_image_list = []
    success_flag_list = []
    for bottom_image, top_image, success_flag in tqdm(result):
        bottom_image_list.append(bottom_image)
        top_image_list.append(top_image)
        success_flag_list.append(success_flag)

    # add lists to dataset
    dataset = dataset.add_column(name="bottom_image", column=bottom_image_list)
    print("[red]Bottom image added[/red]")
    dataset = dataset.add_column(name="top_image", column=top_image_list)
    print("[red]Top image added[/red]")
    dataset = dataset.add_column(name="success_flag", column=success_flag_list)
    print("[red]Success flag added[/red]")
    dataset = dataset.filter(lambda example: example["success_flag"] == True)
    dataset = dataset.remove_columns(["success_flag"])

    # process data
    clip_processor = CLIPProcessor.from_pretrained(
        pretrained_model_name_or_path=model_name
    )

    # process bottom data
    dataset = dataset.map(clip_processor.tokenizer, input_columns="bottom_text", fn_kwargs={"truncation": True, "padding": True, "return_tensors": "pt"}, batched=True, num_proc=num_proc)  # type: ignore
    dataset = dataset.rename_column("input_ids", "bottom_input_ids")
    dataset = dataset.rename_column("attention_mask", "bottom_attention_mask")
    dataset = dataset.map(lambda x: clip_processor.image_processor(np.array(x), return_tensors="pt"), input_columns="bottom_image", batched=True, num_proc=num_proc)  # type: ignore
    dataset = dataset.rename_column("pixel_values", "bottom_processed_image")
    dataset = dataset.map(
        clip_processor,
        input_columns="top_text",
        batched=True,
        fn_kwargs={"truncation": True, "padding": True, "return_tensors": "pt"},
    )
    dataset = dataset.rename_column("input_ids", "top_input_ids")
    dataset = dataset.rename_column("attention_mask", "top_attention_mask")
    dataset = dataset.map(lambda x: clip_processor.image_processor(np.array(x), return_tensors="pt"), input_columns="top_image", batched=True, num_proc=num_proc)  # type: ignore
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
