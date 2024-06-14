import pika
import json
from time import sleep
from ai_package_for_com.exceptions import *


QUEUE_INPUT = 'queue_input'
QUEUE_OUTPUT = 'queue_output'
CONNEXION_URI = 'localhost'


def get_rabbitmq_handle(connection_string, max_retries=3):
    """
    :param connection_string: chaine de connexion à RabbitMQ
    :return: handle de RabbitMQ
    :param max_retries: nombre maximal de tentatives de connexion
    """
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(connection_string))
        channel = connection.channel()
        print('Connexion à RabbitMQ établie')
        return channel, connection
    
    except Exception as e:
        print('Erreur lors de la connexion à RabbitMQ: ', str(e))
        if max_retries > 0:
            sleep(5)
            return get_rabbitmq_handle(connection_string, max_retries - 1)
        else:
            raise MaxAttemptsExceededError('Nombre maximal de tentatives de connexion atteint')


def close_rabbitmq_handle(channel, connection, max_retries=3):
    """
    :param channel: channel de communication avec RabbitMQ
    :param connection: connexion à RabbitMQ
    :param max_retries: nombre maximal de tentatives de fermeture de la connexion
    """
    try:
        channel.close()
        connection.close()
        print('Connexion à RabbitMQ fermée')
        
    except Exception as e:
        print('Erreur lors de la fermeture de la connexion à RabbitMQ: ', str(e))
        if max_retries > 0:
            sleep(5)
            close_rabbitmq_handle(channel, connection, max_retries - 1)
        else:
            raise MaxAttemptsExceededError('Nombre maximal de tentatives de connexion atteint')


class RabbitMQHandler:

    def __init__(self):
        self.queue_in = None

        try:
            self.channel, self.connexion = get_rabbitmq_handle(CONNEXION_URI)
        except MaxAttemptsExceededError as e:
            print(e)
            print("Pour augmenter le nombre de tentatives, modifier la valeur de max_retries dans le code")
            exit(1)
        except Exception as e:
            print('Erreur lors de la connexion à RabbitMQ: ', str(e))
            exit(1)

        try:
            self.queue_in = self.channel.queue_declare(queue=QUEUE_INPUT, durable=False)
            self.channel.queue_declare(queue=QUEUE_OUTPUT, durable=False)
            print('Initialisation de la queue de messages')

        except Exception as e:
            print('Erreur lors de l\'initialisation de la queue de messages: ', str(e))
            exit(1)

    def get_last_request(self, max_retries=3):
        """
        :return: dernière requête reçue
        :param max_retries: nombre maximal de tentatives de réception
        """

        try:
            method_frame, header_frame, body_bytes = self.channel.basic_get(queue=QUEUE_INPUT, auto_ack=True)

            if method_frame:
                body = json.loads(body_bytes.decode('utf-8'))
                print(" [<] Message received: ", body)
                return body

        except Exception as e:
            print('Erreur lors de la réception du message: ', str(e))
            if max_retries > 0:
                sleep(5)
                return self.get_last_request(max_retries - 1)
            raise MessageReceptionError(f'Erreur lors de la réception du message: {e}')

    def send_result(self, body: dict, max_retries=3):
        """
        :param body: contenu du message à envoyer dans la queue de retour
        :param max_retries: nombre maximal de tentatives d'envoi
        """
        try:
            body_bytes = json.dumps(body, default=str).encode('utf-8')

            self.channel.queue_declare(queue=QUEUE_OUTPUT, durable=False, passive=True)
            self.channel.basic_publish(exchange='',
                                       routing_key=QUEUE_OUTPUT,
                                       body=body_bytes)
            print(" [>] Message sent: ", body)
        except Exception as e:
            print('Erreur lors de l\'envoi du message: ', str(e))
            if max_retries > 0:
                sleep(5)
                self.send_result(body, max_retries - 1)
            else:
                raise Exception('Nombre maximal de tentatives d\'envoi atteint')

    def close(self):
        close_rabbitmq_handle(self.channel, self.connexion)
