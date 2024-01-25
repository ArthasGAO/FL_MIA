import logging
import os
import sys
from datetime import datetime

from util import distribute_data_equally, average_weights, parse_experiment_file, dataset_model_set, create_nodes_list, \
    create_adjacency
from lightning import Trainer, seed_everything
import copy
import json
import torch
import Node


def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""
    handler = logging.FileHandler(log_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    handler.setFormatter(formatter)

    return logger


def main():
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Directory and log file for data distribution logging
    data_directory = f'./logs/data_inspection/{current_time}'
    os.makedirs(data_directory, exist_ok=True)
    data_log_file = os.path.join(data_directory, 'data_distribution.log')
    data_logger = setup_logger('data_logger', data_log_file)

    # Directory and log file for model results logging
    model_directory = f'./logs/model_result/{current_time}'
    os.makedirs(model_directory, exist_ok=True)
    model_log_file = os.path.join(model_directory, 'model_result.log')
    model_logger = setup_logger('model_logger', model_log_file)

    COMMAND = f'./command/{sys.argv[1]}.txt'
    with open(COMMAND, 'r') as file:
        for line in file:
            model_logger.info(line.strip())

    experiment_details = parse_experiment_file(COMMAND)
    TOPOLOGY = int(experiment_details["Topology"])
    ROUND = int(experiment_details["Round"])
    NUM_CLIENTS = int(experiment_details["Num_clients"])
    DATASET = experiment_details["Dataset"]
    MODEL = experiment_details["Model"]
    IID = experiment_details["iid"]
    ATTACK = experiment_details["Attack"]
    BATCH_SIZE = int(experiment_details["batch"])
    MAX_EPOCHS = int(experiment_details["max_epochs"])

    seed_everything(42)
    # dataset and model setting
    global_dataset, global_model = dataset_model_set(DATASET, MODEL, data_logger)

    # separate client's dataset: # A list containing all dataloaders
    if int(IID) == 0:
        train_dataloaders = distribute_data_equally(global_dataset.train_set, num_clients=NUM_CLIENTS,
                                                    batch_size=BATCH_SIZE,
                                                    sign=True, iid=IID, logger=data_logger)
        test_dataloaders = distribute_data_equally(global_dataset.test_set, num_clients=NUM_CLIENTS,
                                                   batch_size=BATCH_SIZE,
                                                   sign=False, iid=IID, logger=data_logger)
        val_dataloaders = distribute_data_equally(global_dataset.val_set, num_clients=NUM_CLIENTS,
                                                  batch_size=BATCH_SIZE,
                                                  sign=False, iid=IID, logger=data_logger)

    # nodes setting
    nodes_list = create_nodes_list(NUM_CLIENTS, train_dataloaders, val_dataloaders, test_dataloaders, global_model)
    nodes_list = create_adjacency(nodes_list, TOPOLOGY)

    # trainer setting
    for r in range(ROUND):
        model_logger.info(f"this is the {r + 1} round.")
        # training process
        for client in nodes_list:
            local_trainer = Trainer(max_epochs=MAX_EPOCHS, accelerator="auto", devices="auto", logger=False)
            local_trainer.fit(client.model, client.train_dataloader, client.val_dataloader)
            client.set_current_params(client.model.state_dict())  # store the current trained model params
            train_result = local_trainer.callback_metrics

            local_trainer.test(client.model, client.test_dataloader, verbose=False)
            test_result = local_trainer.callback_metrics

            merged_results = {**train_result, **test_result}
            res_dict = {key: value.item() if hasattr(value, 'item') else value for key, value in merged_results.items()}
            model_logger.info(f"{client.idx}'s {r + 1} round model result: {json.dumps(res_dict, indent=None)}")
            model_logger.info('')
        # Aggregation process
        for client in nodes_list:
            client.aggregate_weights()

    torch.save(global_model.state_dict(), f"./final_model/global_model_{sys.argv[1]}.pth")


if __name__ == '__main__':
    main()
