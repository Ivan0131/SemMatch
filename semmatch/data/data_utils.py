import os
import tensorflow as tf
from semmatch.utils.logger import logger
import multiprocessing
import zipfile
import requests
from urllib.error import HTTPError
from urllib.request import urlretrieve
import nltk


def cpu_count():
  """Return the number of available cores."""
  num_available_cores = multiprocessing.cpu_count()
  return num_available_cores


def download_report_hook(count, block_size, total_size):
    """Report hook for download progress.

    Args:
    count: current block number
    block_size: block size
    total_size: total size
    """
    percent = int(count * block_size * 100 / total_size)
    print("\r%d%%" % percent + " completed", end="\r")


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    if os.path.exists(destination):
        print("File {} exists, skipping ...".format(destination))
        return

    print("Downloading {} ...".format(destination))
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def unzip(file_name, file_dir):
    with zipfile.ZipFile(file_name) as archive:
        if not all(os.path.exists(os.path.join(file_dir, name)) for name in archive.namelist()):
            logger.info("Extracting: " + file_name)
            archive.extractall(file_dir)
        # else:
        #     print("Content exists, extraction skipped.")


def maybe_download(filepath, url):
    """Download filename from url unless it's already in directory.

    Copies a remote file to local if that local file does not already exist.  If
    the local file pre-exists this function call, it does not check that the local
    file is a copy of the remote.

    Args:
      directory: path to the directory that will be used.
      filename: name of the file to download to (do nothing if it already exists).
      url: URL to copy (or download) from.

    Returns:
      The path to the downloaded file.
    """

    if os.path.exists(filepath):
        logger.info("Not downloading, file already found: %s" % filepath)
        return filepath

    logger.info("Downloading %s to %s" % (url, filepath))
    try:
        tf.gfile.Copy(url, filepath)
    except tf.errors.UnimplementedError:
        try:
            inprogress_filepath = filepath + ".incomplete"
            # r = requests.get(url)
            # with open(inprogress_filepath, 'wb') as outfile:
            #     outfile.write(r.content)

            inprogress_filepath, _ = urlretrieve(
                url, inprogress_filepath, reporthook=download_report_hook)
            # Print newline to clear the carriage return from the download progress
            print()
            os.rename(inprogress_filepath, filepath)
        except HTTPError:
            if url.startswith("http"):
                os.system('wget --no-check-certificat ' + url+" -O "+filepath.replace(" ", "\ "))

            else:
                raise ValueError("Unrecognized URI: " + filepath)
    statinfo = os.stat(filepath)
    logger.info("Successfully downloaded %s, %s bytes." %
                    (os.path.basename(filepath), statinfo.st_size))
    return filepath


def get_exact_match(premise, hypothesis):
    stemmer = nltk.stem.LancasterStemmer()
    premise = [stemmer.stem(token.text) for token in premise]
    hypothesis = [stemmer.stem(token.text) for token in hypothesis]
    premise_set = set(premise)
    hypothesis_set = set(hypothesis)
    premise_exact_match = []
    for i, token in enumerate(premise):
        if token in hypothesis_set:
            premise_exact_match.append(i)

    hypothesis_exact_match = []
    for i, token in enumerate(hypothesis):
        if token in premise_set:
            hypothesis_exact_match.append(i)

    return premise_exact_match, hypothesis_exact_match