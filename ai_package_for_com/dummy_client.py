import pika
from time import sleep
import json


QUEUE_INPUT = 'queue_input'
QUEUE_OUTPUT = 'queue_output'
CONNEXION_URI = 'localhost'


def get_rabbitmq_handle(connection_string):
    """
    :param connection_string: chaine de connexion à RabbitMQ
    :return: handle de RabbitMQ
    """
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(connection_string))
        channel = connection.channel()
        print('Connexion à RabbitMQ établie')
        return channel, connection

    except Exception as e:
        print('Erreur lors de la connexion à RabbitMQ: ' + str(e))
        exit(1)


def send_loop():
    channel, connection = get_rabbitmq_handle(connection_string=CONNEXION_URI)
    channel.queue_declare(queue=QUEUE_INPUT)

    while True:
        try:
            data = {
                "socket_id": "oui",
                "id": 6,
                "message": "Qu'est ce qu'une attaque DDOS ?"
            }
            body_bytes = json.dumps(data).encode('utf-8')
            channel.basic_publish(exchange='',
                                  routing_key=QUEUE_INPUT,
                                  body=body_bytes)
            print(" [>] Message sent")
            sleep(10)

        except KeyboardInterrupt:
            print('Interruption clavier')
            connection.close()
            channel.close()
            exit(0)


if __name__ == "__main__":
    send_loop()
