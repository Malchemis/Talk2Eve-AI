# Talk2Eve - The AI part of our pipeline
Talk2Eve is a Python based chatbot that revolves around using the [MITRE ATT&CK](https://attack.mitre.org/) knowledge base.

## Installation
1. Clone the repository
2. Install the required packages. We recommend using a conda environment.
```bash
conda create -n talk2eve python=3.11
```
3. Install the required packages
```bash
bash requirements.sh
```
The script will download the required packages, as well as clone the [Vigogne](https://github.com/bofenghuang/vigogne.git) repository.
This is used for personalization of the chatbot.

## Usage
1. Define you configuration parameters in the `config.yaml` file.
2. You may have to take a look at the `db_handler` and `rabbitq_handler` depending on you MongoDB and RabbitMQ configuration.
3. Run the `main.py` file. 

Note : You can modify the level of the logging to see the debugging or error messages, if any.
Don't hesitate to use your own. 

The archive folder was kept for reference, and contains some of the research done on the models and the knowledge base. 