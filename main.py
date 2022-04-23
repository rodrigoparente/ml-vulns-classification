# python imports
import shutil

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
    shutil.rmtree('logs')
    shutil.rmtree('results')

    # execute tests
    execute()

    # process tests
    process()
