from rabbitmq.rabbitmq_handler import RabbitMQHandler
from mongoDB.db_handler import DBHandler
from ia.dummy_ai import DummyAI
from exceptions import *
from time import sleep
import sys


REFRESH_DELAY = 2


def main_loop():
    rabbitmq = RabbitMQHandler()
    ia = DummyAI()
    db = DBHandler()

    while True:
        try:
            print('Attente de requête...')
            last_req = None

            while last_req is None:
                last_req = rabbitmq.get_last_request()
                print('Queue vide')
                sleep(REFRESH_DELAY)

            print('IA en cours de traitement...')
            try:
                context = db.findOne({'id': last_req['id']})['context']

            except Exception as e:
                print(e)
                print('Erreur lors de la récupération du contexte')
                context = []
            
            prompt = {
                'role': 'user',
                'content': last_req['message']
            }

            context.append(prompt)
            if len(context) > 5:
                context = context[-5:]

            db.update({'id': last_req['id']}, 'context', context)
            res = ia.chat(socket_id=last_req['socket_id'], context=context)

            print('Envoi du résultat...')
            rabbitmq.send_result(res)

        except MessageReceptionError as e:
            print(e)
        except Exception as e:
            print(e)
            print("Arret du serveur")
            rabbitmq.close()
            sys.exit(1)
        except KeyboardInterrupt:
            print('Interruption clavier')
            rabbitmq.close()
            sys.exit(0)


if __name__ == "__main__":
    main_loop()