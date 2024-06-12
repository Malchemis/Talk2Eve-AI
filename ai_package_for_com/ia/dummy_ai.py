from time import sleep


class DummyAI:

    def chat(self, socket_id, context):
        sleep(8)
        return {
            "socket_id": socket_id,
            "response": f"Requete traitée: {context[-1]['content']}"
        }