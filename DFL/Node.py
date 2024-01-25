import copy
import torch


class Node:
    def __init__(self, idx, model, train_dataloader, val_dataloader, test_dataloader):
        self.idx = idx
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.neigh = set()
        self.current_params = None

    def aggregate_weights(self):
        local_weights = copy.deepcopy(self.model.state_dict())

        for key in local_weights.keys():
            for node in self.neigh:
                local_weights[key] += node.current_params[key]
            local_weights[key] = torch.div(local_weights[key], len(self.neigh) + 1)

        self.model.load_state_dict(local_weights)

    def set_current_params(self, params):
        self.current_params = params
