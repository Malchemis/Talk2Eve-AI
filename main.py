import traceback
import yaml
import time

from utils import set_num_threads, print_chat, MyTextStreamer
from chatbob import ChatHandler


def main(config):
    set_num_threads(config['torch']['num_workers'])
    chatBob = ChatHandler(**config['chat_handler'])
    chat_history = []
    while True:
        user_prompt = input("Enter your query: ")
        if user_prompt == 'exit':
            break
        start = time.time()
        try:
            chat_history.append({'role': 'user', 'content': user_prompt})
            chat_history = chatBob.chat(chat_history)
        except:
            traceback.print_exc()
            continue
        print_chat(chat_history)
        print('\nObtained in', time.time() - start, 'seconds')


if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        config_file = yaml.safe_load(file)
    main(config_file)
