#!/usr/bin/env python3
import os
from importlib import import_module
from termcolor import colored
import argparse
import torch
import numpy as np

from coffail_utils.policies import policy_modules
from coffail_utils.dataset.policy_learning_dataset import PolicyLearningDataset

'''
A demo script for training a vision transformer-based policy
for one of the COFFAIL skills using imitation learning.
The following is an example call of the script:

python3 imitation_policy_learner.py \
    --policy-type vit-vmp \
    --robot jessie \
    --skill pickup-cup \
    --image-observation-key head_camera_image \
    --model-path /path/to/jessie/pickup-cup \
    --epochs 10 \
    --batch-size 16 \
    --learning-rate 0.00001
'''

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-pt', '--policy-type', type=str,
                           choices=['vit-vmp'],
                           help='Type of the policy to be learned')
    argparser.add_argument('-r', '--robot', type=str,
                           help='Robot name')
    argparser.add_argument('-s', '--skill', type=str,
                           help='Skill name')
    argparser.add_argument('-obs', '--image-observation-key', type=str,
                           help='Image observation keys')
    argparser.add_argument('-m', '--model-path', type=str,
                           help='Directory where pretrained model snapshots are saved')
    argparser.add_argument('-ep', '--epochs', type=int,
                           help='Number of training epochs')
    argparser.add_argument('-b', '--batch-size', type=int,
                           help='Training batch size')
    argparser.add_argument('-lr', '--learning-rate', type=float,
                           help='Initial learning rate for the optimiser')

    args = argparser.parse_args()
    policy_type = args.policy_type
    robot_name = args.robot
    skill_name = args.skill
    image_observation_key = args.image_observation_key
    model_path = args.model_path
    number_of_epochs = args.epochs
    training_batch_size = args.batch_size
    learning_rate = args.learning_rate

    print(colored('----------------------------------', 'green'))
    print(colored('Received the following parameters:', 'green'))
    print(colored('----------------------------------', 'green'))
    print(colored(f'policy type: {policy_type}', 'green'))
    print(colored(f'robot: {robot_name}', 'green'))
    print(colored(f'skill: {skill_name}', 'green'))
    print(colored(f'image observation: {image_observation_key}', 'green'))
    print(colored(f'model path: {model_path}', 'green'))
    print(colored(f'number of epochs: {number_of_epochs}', 'green'))
    print(colored(f'batch size: {training_batch_size}', 'green'))
    print(colored(f'learning rate: {learning_rate}', 'green'))
    print(colored('----------------------------------', 'green'))

    policy_module, policy_class_name = policy_modules[policy_type]
    PolicyClass = getattr(import_module(policy_module), policy_class_name)
    policy = PolicyClass(number_of_actions=6)

    dataset_train = PolicyLearningDataset(robot_name=robot_name,
                                          observation_keys=[image_observation_key],
                                          skill_name=skill_name)
    optim_criterion = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(policy.policy_net.parameters(), lr=learning_rate)
    data_loader = torch.utils.data.DataLoader(dataset_train,
                                              batch_size=training_batch_size,
                                              num_workers=1)

    losses = []
    for epoch in range(number_of_epochs):
        running_loss = 0.
        minibatch_counter = 0
        for _, data in enumerate(data_loader, 0):
            observations, actions, _ = data
            optimiser.zero_grad()

            outputs = policy.act(image=observations[image_observation_key])
            loss = optim_criterion(outputs.float(), actions.float())
            loss.backward()
            optimiser.step()

            minibatch_counter += 1
            running_loss += loss.item()
        epoch_average_loss = running_loss / minibatch_counter
        losses.append(epoch_average_loss)
        print(colored(f'Epoch: {epoch+1}; average loss: {epoch_average_loss}', 'green'))

        model_checkpoint_name = os.path.join(model_path, skill_name + f'-ep{epoch}.pt')
        policy.save(model_checkpoint_name)
        print(colored(f'Checkpoint saved to {model_checkpoint_name}', 'green'))

    losses_file_name = os.path.join(model_path, skill_name + f'-losses.txt')
    print(colored(f'Losses saved to {losses_file_name}', 'green'))
    np.savetxt(losses_file_name, np.array(losses))

    print(colored('Finished training', 'green'))
