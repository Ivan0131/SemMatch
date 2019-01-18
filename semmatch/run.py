import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
import logging
from semmatch import main


logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":
    main(prog="semmatch")