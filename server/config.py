# encoding: utf-8

import os
import yaml


def load_config(config_file):
    """
    读取配置
    :param config_file:
    :return:
    """
    with open(config_file, 'r', encoding='utf8') as fd:
        try:
            return yaml.safe_load(fd)
        except yaml.YAMLError as exc:
            print(exc)


ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
CONFIG_PATH = os.path.join(ROOT_PATH, "conf/config.yaml")
CONFIG = load_config(CONFIG_PATH)

