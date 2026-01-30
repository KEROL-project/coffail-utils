import abc
import os
import numpy as np
import torch
import pymongo as pm

class MongoDataset(torch.utils.data.IterableDataset):
    def __init__(self, robot_name: str,
                 observation_keys: dict[str, str],
                 skill_name: str=''):
        super(MongoDataset).__init__()
        self.robot_name = robot_name
        self.observation_keys = observation_keys
        db_client = MongoDataset.get_db_client()
        database = db_client[robot_name]
        self.data_collection_names = database.list_collection_names()

        # we filter only the collection names that correspond to the specified skill
        if skill_name:
            self.data_collection_names = [name for name in self.data_collection_names
                                          if skill_name in name]

    def __iter__(self):
        return iter(self.get_next_item())

    @abc.abstractmethod
    def get_next_item(self):
        raise NotImplementedError('get_next_item not implemented')

    @staticmethod
    def get_db_client() -> pm.MongoClient:
        '''Returns a MongoDB client at <host>:<port>. By default,
        <host> is "localhost" and <port> is 27017, but these values can
        be overriden by setting the environment variables "DB_HOST" and
        "DB_PORT" respectively.

        Method taken from https://github.com/ropod-project/black-box-tools/blob/master/black_box_tools/db_utils.py
        '''
        (host, port) = MongoDataset.get_db_host_and_port()
        client = pm.MongoClient(host=host, port=port)
        return client

    @staticmethod
    def get_db_host_and_port() -> tuple[str, int]:
        '''Returns a (host, port) tuple which is ("localhost", 27017) by default,
        but the values can be overridden by setting the environment variables
        "DB_HOST" and "DB_PORT" respectively.

        Method taken from https://github.com/ropod-project/black-box-tools/blob/master/black_box_tools/db_utils.py
        '''
        host = 'localhost'
        port = 27017
        if 'DB_HOST' in os.environ:
            host = os.environ['DB_HOST']
        if 'DB_PORT' in os.environ:
            port = int(os.environ['DB_PORT'])
        return (host, port)

