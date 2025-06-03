import argparse
import os
import tarfile
import urllib.request

from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def post_process_dataset(dataset_path: str):
    for table in os.listdir(dataset_path):
        os.rename(
            f"{dataset_path}/{table}", f"{dataset_path}/{table.split('_SEP_')[1]}"
        )
    print("Processed the dataset")


def extract_tar(tar_name: str, extract_path="."):
    try:
        with tarfile.open(tar_name, "r") as tar:
            tar.extractall(path=extract_path, filter="data")
            print(f"Extracted all files to '{extract_path}'")
        os.remove(tar_name)
        print(f"Removed the tar file: '{tar_name}'")
    except Exception as e:
        print(f"An error occurred: {e}")


def download_with_progress(url, filename):
    def reporthook(block_num, block_size, total_size):
        if reporthook.pbar.total is None:
            reporthook.pbar.total = total_size
            reporthook.pbar.reset(total=total_size)
        downloaded = block_num * block_size
        reporthook.pbar.update(downloaded - reporthook.pbar.n)

    reporthook.pbar = tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc=filename)
    urllib.request.urlretrieve(url, filename, reporthook)
    reporthook.pbar.close()


dataset_mappings = {
    "public_bi": {"id": "4ttqhyqxopjl09s6jlnltpa0p7uzw4xo", "filename": "pneuma_public_bi.tar"},
    "chicago": {"id": "hby2hf60rlmlbrlkloapbusuxb94kavy", "filename": "pneuma_chicago_10K.tar"},
    "chembl": {"id": "qulaggkktes5f2tj04hb128a7r6uh2ix", "filename": "pneuma_chembl_10K.tar"},
    "fetaqa": {"id": "859o2svdh5owo4vf92iuo3t72pc67mmw", "filename": "pneuma_fetaqa.tar"},
    "adventure": {"id": "w8a8ro0jyw583aq0uh8cgo0tzgm4vqw5", "filename": "pneuma_adventure_works.tar"},
    "bird": {"id": "https://storage.googleapis.com/pneuma_open/pneuma_bird.tar", "filename": "pneuma_bird.tar"},  # Future-TODO: Migrate to Box
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This program downloads datasets used in the experiments.",
    )
    parser.add_argument("-d", "--dataset", default="all")
    dataset = parser.parse_args().dataset

    if dataset == "all":
        for _, dataset_mapping in tqdm(dataset_mappings.items()):
            dataset_id = dataset_mapping['id']
            dataset_filename = dataset_mapping['filename']
            if dataset_id.startswith('http'):
                download_with_progress(
                    dataset_id,
                    filename=os.path.join(SCRIPT_DIR, dataset_filename),
                )
            else:
                download_with_progress(
                    f"https://uchicago.box.com/shared/static/{dataset_id}",
                    filename=os.path.join(SCRIPT_DIR, dataset_filename),
                )
            extract_tar(os.path.join(SCRIPT_DIR, dataset_filename), SCRIPT_DIR)
            post_process_dataset(os.path.join(SCRIPT_DIR, dataset_filename[:-4]))
    else:
        try:
            dataset_id = dataset_mappings[dataset]['id']
            dataset_filename = dataset_mappings[dataset]['filename']
            if dataset_id.startswith('http'):
                download_with_progress(
                    dataset_id,
                    filename=os.path.join(SCRIPT_DIR, dataset_filename),
                )
            else:
                download_with_progress(
                    f"https://uchicago.box.com/shared/static/{dataset_id}",
                    filename=os.path.join(SCRIPT_DIR, dataset_filename),
                )
            extract_tar(os.path.join(SCRIPT_DIR, dataset_filename), SCRIPT_DIR)
            post_process_dataset(os.path.join(SCRIPT_DIR, dataset_filename[:-4]))
        except KeyError:
            print(
                f"Dataset {dataset} not found!"
            )
