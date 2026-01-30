import abc
import numpy as np
import torch
import pymongo as pm

from coffail_utils.dataset.mongo_dataset import MongoDataset

class PolicyLearningDataset(MongoDataset):
    def __init__(self, robot_name, observation_keys, skill_name=''):
        super(PolicyLearningDataset, self).__init__(robot_name=robot_name,
                                                    observation_keys=observation_keys,
                                                    skill_name=skill_name)

    def get_next_item(self):
        for collection_name in self.data_collection_names:
            db_client = MongoDataset.get_db_client()
            database = db_client[self.robot_name]
            collection = database[collection_name]
            collection_docs = collection.find()
            for doc in collection_docs:
                # we skip steps for which no action has been performed
                linear_action_norm = np.linalg.norm(doc['action'][0:3])
                angular_action_norm = np.linalg.norm(doc['action'][3:])
                if linear_action_norm < 1e-5 and angular_action_norm < 1e-5:
                    continue

                observations = {}
                for obs_name in self.observation_keys:
                    observations[obs_name] = torch.Tensor(doc['observation'][obs_name]).permute(2, 0, 1)
                action = torch.Tensor(doc['action'])
                info = {'first_episode_step': doc['step_id'] == 0}
                yield (observations, action, info)

