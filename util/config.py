import json
from typing import List


class ActorConfig:
    def __init__(self):
        self.host = ''
        self.port = 0
        self.total_steps = 0

    def load_dict(self, config_dict: dict):
        self.host = config_dict.get('host', 'localhost')
        self.port = config_dict.get('port', 8080)
        self.total_steps = config_dict.get('total_steps', 0)

    def __str__(self):
        return '{}:{} / steps: {}'.format(self.host, self.port, self.total_steps)


class LearnerConfig:
    def __init__(self):
        self.total_steps = 0
        self.gpu_option = 0

    def load_dict(self, config_dict: dict):
        self.total_steps = config_dict.get('total_steps', 10)
        self.gpu_option = config_dict.get('gpu_option', 0)

    def __str__(self):
        return 'GPU {} / steps: {}'.format(self.gpu_option, self.total_steps)


def read_actors(json_file) -> List[ActorConfig]:
    with open(json_file) as f_in:
        conf_list = json.load(f_in)
    actor_list = []
    for v in conf_list:
        conf = ActorConfig()
        conf.load_dict(v)
        actor_list.append(conf)

    return actor_list


def read_learner(yaml_file) -> LearnerConfig:
    with open(yaml_file) as f_in:
        conf_dict = json.load(f_in)
    conf = LearnerConfig()
    conf.load_dict(conf_dict)
    return conf
