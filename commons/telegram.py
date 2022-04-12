# python imports
import os

# third-party imports
import requests


def send_message(msg):

    chat_id = os.environ.get('CHAT_ID', None)
    bot_id = os.environ.get('BOT_ID', None)

    if chat_id is None or bot_id is None:
        return

    data_post = f'{{"chat_id": "{chat_id}", "text": "{msg}", "disable_notification": false"}}'

    requests.post(
        f'https://api.telegram.org/{bot_id}/sendMessage',
        data=data_post, headers={"Content-Type": "application/json"})
