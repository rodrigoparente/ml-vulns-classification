# python imports
import shutil
import os

# third-party imports
from dotenv import load_dotenv

# local imports
from execute import execute
from process import process


if __name__ == '__main__':
    # take environment
    # variables from .env
    load_dotenv()

    # clean results & logs
    for dir in ['logs', 'results']:
        if os.path.exists(dir):
            shutil.rmtree(dir)

    # execute tests
    execute()

    # process tests
    process()
