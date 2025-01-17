import os
import argparse

import gdown

from utils.paths import DATASET_DIR

# registry of datasets to download (name -> file_id)
DATASET_REGISTRY = {
    "Panda_Panda_TwoArmLift_lowdim": "1Q6GCdZurAnSbd_TJPmOO_LUw8lm6pW4i"
}

def download_file_from_google_drive(file_id: str, destination: str):
    """
    Downloads a file from Google Drive.

    Args:
        file_id (str): Google Drive file ID
        destination (str): file to write
    """
    download_url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(download_url, destination, quiet=False)


def download_dataset(dataset_name: str, download_dir: str = DATASET_DIR):
    """
    Downloads the dataset with name @dataset_name to the directory @save_dir.

    Args:
        dataset_name (str): name of dataset to download
        download_dir (str): directory to save dataset to
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError("Dataset {} not found in registry.".format(dataset_name))

    file_id = DATASET_REGISTRY[dataset_name]

    os.makedirs(download_dir, exist_ok=True)
    print("Downloading dataset {} to {}".format(dataset_name, download_dir))
    download_file_from_google_drive(
        file_id=file_id,
        destination=os.path.join(download_dir, dataset_name + ".hdf5")
    )

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Download all datasets."
    )

    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="",
        help="Download a specific dataset."
    )

    args = parser.parse_args()

    assert not (args.all and args.dataset != ""), "Cannot specify both --all and --dataset."
    assert args.all or args.dataset != "", "Must specify either --all or --dataset."

    if args.all:
        for dataset_name in DATASET_REGISTRY:
            download_dataset(dataset_name)

    else:
        if not args.dataset in DATASET_REGISTRY:
            print("Dataset {} not found in registry. Available datasets: {}".format(
                args.dataset, list(DATASET_REGISTRY.keys())
            ))
            return
        download_dataset(args.dataset)

    print("Download complete.")

if __name__ == "__main__":
    main()

