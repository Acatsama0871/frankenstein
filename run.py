import os
import typer
from src import process_data_dataset, compose_config


app = typer.Typer()


@app.command("process-data")
def process_data_func(
    text_file_path: str = typer.Option(
        os.path.join("data", "01_raw", "new_outfits_merged_texts_dicts_lst_data.json"),
        "-tfp",
        "--text-file-path",
    ),
    img_root_folder_path: str = typer.Option(
        os.path.join("data", "01_raw", "outfits"), "-irfp", "--img-root-folder-path"
    ),
    save_path: str = typer.Option(
        os.path.join("data", "02_intermediate", "evaluation3"), "-sp", "--save-path"
    ),
    config_name: str = typer.Option("process_data", "-cn", "--config-name"),
):
    config = compose_config(config_name=config_name)
    process_data_dataset(
        text_file_path=text_file_path,
        img_root_folder_path=img_root_folder_path,
        save_path=save_path,
        config=config,
    )


if __name__ == "__main__":
    app()
