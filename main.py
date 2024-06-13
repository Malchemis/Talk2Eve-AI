# Description: Main file for the chatbot server
import yaml
import logging

from utils import set_num_threads
from chatbob import ChatHandler

from ai_package_for_com.rabbitmq.rabbitmq_handler import RabbitMQHandler
from ai_package_for_com.mongoDB.db_handler import DBHandler
from ai_package_for_com.exceptions import *
from time import sleep
import sys


def main_loop(config, logger):
    set_num_threads(config['torch']['num_workers'])
    rabbitmq = RabbitMQHandler()
    db = DBHandler()
    ia = ChatHandler(queue=rabbitmq, logger=logger, **config['chat_handler'])

    th = config['max_conv_length']
    ref_delay = config['refresh_delay']
    while True:
        try:
            logger.debug('Attente de requête...')
            last_req = None

            while last_req is None:
                last_req = rabbitmq.get_last_request()
                logger.debug('Queue vide')
                sleep(ref_delay)

            logger.debug('IA en cours de traitement...')
            try:
                chat_history = db.findOne({'id': last_req['id']})['context']
            except Exception as e:
                logger.debug(e)
                logger.debug('Erreur lors de la récupération du contexte')
                chat_history = []

            prompt = {'role': 'user', 'content': last_req['message']}

            chat_history.append(prompt)
            if len(chat_history) > th:
                chat_history = chat_history[-th:]
            try:
                chat_history = ia.chat(chat_history, last_req['socket_id'], last_req['id'])
                db.update({'id': last_req['id']}, 'context', chat_history)
                logger.info(chat_history)

            except UpdateContextException as e:
                logger.error(e)
            except Exception as e:
                logger.error(e)
                response = {"status": "error", "socket_id": last_req['socket_id'],
                            "acccess_token": last_req['id'],
                            "error": "An error occurred, please try again"}
                rabbitmq.send_result(response)

        except MessageReceptionError as e:
            logger.error(e)
        except Exception as e:
            logger.error(e)
            logger.error("Arret du serveur")
            rabbitmq.close()
            sys.exit(1)
        except KeyboardInterrupt:
            logger.error('Interruption clavier')
            rabbitmq.close()
            sys.exit(0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    with open('config.yaml', 'r') as file:
        config_file = yaml.safe_load(file)
    main_loop(config_file, logger=logging.getLogger('Talk2Eve'))
