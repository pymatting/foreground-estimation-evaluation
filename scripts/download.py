import os, util, zipfile
from urllib.request import urlopen


def download(download_directory):
    os.makedirs(download_directory, exist_ok=True)

    urls = [
        "http://www.alphamatting.com/datasets/zip/input_training_lowres.zip",
        "http://www.alphamatting.com/datasets/zip/trimap_training_lowres.zip",
        "http://www.alphamatting.com/datasets/zip/gt_training_lowres.zip",
        "http://www.alphamatting.com/datasets/zip/input_training_highres.zip",
        "http://www.alphamatting.com/datasets/zip/trimap_training_highres.zip",
        "http://www.alphamatting.com/datasets/zip/gt_training_highres.zip",
        "http://www.alphamatting.com/datasets/zip/input_with_gt_fgd.zip",
    ]

    print("Downloading dataset (approx. 2 GB in total)")

    for url in urls:
        filename = os.path.split(url)[-1]

        path = os.path.join(download_directory, filename)

        if os.path.isfile(path):
            print("Already downloaded:", filename)
        else:
            print("Downloading", url)
            with open(path, "wb") as f:
                n_bytes = 0
                with urlopen(url) as r:
                    while True:
                        chunk = r.read(10 ** 6)

                        if len(chunk) == 0:
                            break

                        if len(chunk) < 0:
                            raise Exception("Failed to download", url)

                        f.write(chunk)
                        n_bytes += len(chunk)
                        print(n_bytes * 1e-6, "MB")

        target_dir = os.path.join(download_directory, os.path.splitext(filename)[0])

        if os.path.isdir(target_dir):
            print("Already unzipped:", target_dir)
        else:
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(target_dir)


if __name__ == "__main__":
    download(util.find_data_directory())
