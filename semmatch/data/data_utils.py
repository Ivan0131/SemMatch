import os
import urllib.request as urllib
import tensorflow as tf
from semmatch.utils.logger import logger
import multiprocessing


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


def maybe_download(directory, filename, uri):
    """Download filename from uri unless it's already in directory.

    Copies a remote file to local if that local file does not already exist.  If
    the local file pre-exists this function call, it does not check that the local
    file is a copy of the remote.

    Args:
      directory: path to the directory that will be used.
      filename: name of the file to download to (do nothing if it already exists).
      uri: URI to copy (or download) from.

    Returns:
      The path to the downloaded file.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        logger.info("Not downloading, file already found: %s" % filepath)
        return filepath

    logger.info("Downloading %s to %s" % (uri, filepath))
    try:
        tf.gfile.Copy(uri, filepath)
    except tf.errors.UnimplementedError:
        if uri.startswith("http"):
            inprogress_filepath = filepath + ".incomplete"
            inprogress_filepath, _ = urllib.urlretrieve(
                uri, inprogress_filepath, reporthook=download_report_hook)
            # Print newline to clear the carriage return from the download progress
            print()
            os.rename(inprogress_filepath, filepath)
        else:
            raise ValueError("Unrecognized URI: " + filepath)
    statinfo = os.stat(filepath)
    logger.info("Successfully downloaded %s, %s bytes." %
                    (filename, statinfo.st_size))
    return filepath
