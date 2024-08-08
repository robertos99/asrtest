import os
import time
from pathlib import Path
from typing import List, Dict, Any, Type
import numpy as np
from collections import OrderedDict
import torch
from abc import ABC, abstractmethod
from pytorch_lightning.loggers import TensorBoardLogger
import nemo.collections.asr as nemo_asr
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from omegaconf import open_dict, DictConfig
import pytorch_lightning as ptl
import pickle
from pytorch_lightning import Callback
import hydra

LOGGER = TensorBoardLogger("experiments/lang-en/federated-en-clean", name="federated_learning")

TOTAL_EPOCH = 0


EPOCH_PER_ROUND = 5
DATA_PER_SPEAKER = 10
SPEAKER_PER_CLIENT = 8
ROUNDS = 10
CLIENTS_PER_ROUND = 10

class LogLearningRate(Callback):
    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        global TOTAL_EPOCH, LOGGER

        lr = trainer.optimizers[0].param_groups[0]['lr']
        LOGGER.experiment.add_scalar("learning_rate", scalar_value=lr, global_step=TOTAL_EPOCH)
        TOTAL_EPOCH += 1


def log_parameters(filename, parameters):
    with open(filename, 'wb') as f:
        pickle.dump(parameters, f)

def set_parameters(net: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    state_dict = OrderedDict()
    params_dict = zip(net.state_dict().keys(), parameters)
    for k, v in params_dict:
        tensor_v = torch.tensor(v, dtype=torch.float32)
        state_dict[k] = tensor_v
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_config(net, train_manifest_path, test_manifest_path):
    cfg = net.cfg
    with (open_dict(cfg)):
        cfg.train_ds.manifest_filepath = train_manifest_path
        # cfg.train_ds.batch_size = 1
        cfg.train_ds.batch_size = 5
        cfg.train_ds.num_workers = 8
        cfg.train_ds.is_tarred = False
        #   cfg.train_ds.pin_memory=True \
        cfg.test_ds.manifest_filepath = test_manifest_path
        cfg.test_ds.batch_size = 1
        cfg.test_ds.num_workers = 8

        # Validation dataset  (Use test dataset as validation, since we train using train + dev)
        cfg.validation_ds.manifest_filepath = test_manifest_path
        cfg.validation_ds.batch_size = 8
        cfg.validation_ds.num_workers = 8
        cfg.train_ds.is_tarred = False

        cfg.spec_augment.freq_mask: 3
        cfg.spec_augment.time_mask: 8

        cfg.optim = {}

    net.cfg = net._cfg
    net.setup_training_data(cfg.train_ds)
    net.setup_validation_data(cfg.validation_ds)
    net.setup_optimization(cfg.optim)
    net.spec_augmentation = net.from_config_dict(cfg.spec_augment)


class FedSchedEncEecCtcBpe(nemo_asr.models.EncDecCTCModelBPE):

    def __init__(self, cfg: DictConfig, trainer=None):
        super().__init__(cfg, trainer)
        self.last_epoch = None

    def set_last_epoch(self, last_epoch):
        self.last_epoch = last_epoch

    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        print(f"--- INFO ---  called FedSchedEncEecCtcBpe configure_optimizers")
        lr = 1e-5
        optimizer = AdamW(self.parameters(), lr=lr)

        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = lr

        train_samples = DATA_PER_SPEAKER * SPEAKER_PER_CLIENT
        max_steps = train_samples * EPOCH_PER_ROUND

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_steps,
            eta_min=1e-7,
            last_epoch=self.last_epoch
        )
        return [optimizer], [scheduler]


class FedClient(ABC):

    @abstractmethod
    def set_parameters(self, parameters: List[np.ndarray]):
        pass

    @abstractmethod
    def get_parameters(self) -> List[np.ndarray]:
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def eval(self) -> Dict[str, Any]:
        pass


def merge_manifests(manifest_paths: list, tmp_dir: str) -> str:
    # Create temporary directory if it doesn't exist
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    # Create a unique filename using a timestamp
    timestamp = int(time.time())
    merged_manifest_path = os.path.join(tmp_dir, f'merged_manifest_{timestamp}.json')

    # Combine lines from each manifest file
    with open(merged_manifest_path, 'w') as merged_file:
        for manifest_path in manifest_paths:
            with open(manifest_path, 'r') as f:
                for line in f:
                    merged_file.write(line)

    return merged_manifest_path


def read_speaker_manifests(output_dir):
    speaker_manifest_paths = set()
    for filename in os.listdir(output_dir):
        if filename.endswith("_manifest.json"):
            speaker_manifest_paths.add(os.path.join(output_dir, filename))
    return speaker_manifest_paths


fed_manifests_folder_path = "timit-dataset/federated-manifests"
speaker_manifest_paths = read_speaker_manifests(fed_manifests_folder_path)
speaker_manifest_paths_copy = set(speaker_manifest_paths)

TEST_MANIFEST_PATH = "timit-dataset/test-manifest.clean.json"


class NemoFedClient(FedClient):

    def __init__(self, round: int, log_lr=False):
        global speaker_manifest_paths, speaker_manifest_paths_copy, TEST_MANIFEST_PATH

        num_pops = SPEAKER_PER_CLIENT

        # Ensure the set is refilled if it has fewer items than num_pops
        if len(speaker_manifest_paths_copy) < num_pops:
            speaker_manifest_paths_copy = set(speaker_manifest_paths)

        train_manifest_paths = []

        for _ in range(num_pops):
            train_manifest_paths.append(speaker_manifest_paths_copy.pop())

        tmp_dir = 'tmp-manifest'
        train_manifest_path = merge_manifests(train_manifest_paths, tmp_dir)

        self.model = FedSchedEncEecCtcBpe.from_pretrained(model_name="stt_en_squeezeformer_ctc_xsmall_ls")
        train_manifest_path = train_manifest_path
        set_config(self.model, train_manifest_path, TEST_MANIFEST_PATH)

        epoch_per_round = EPOCH_PER_ROUND
        last_total_epoch = epoch_per_round * (round - 1)
        self.model.set_last_epoch(last_total_epoch)

        callbacks = []

        if log_lr:
            callbacks.append(LogLearningRate())

        if torch.cuda.is_available():
            accelerator = 'gpu'
        else:
            accelerator = 'cpu'
        self.trainer = ptl.Trainer(devices=1,
                                   accelerator=accelerator,
                                   max_epochs=epoch_per_round,
                                   accumulate_grad_batches=1,
                                   enable_checkpointing=False,
                                   logger=False,
                                   log_every_n_steps=1,
                                   check_val_every_n_epoch=epoch_per_round + 1,
                                   callbacks=callbacks)
        self.model.set_trainer(self.trainer)

    def set_parameters(self, parameters: List[np.ndarray]):
        set_parameters(self.model, parameters)

    def get_parameters(self) -> List[np.ndarray]:
        return get_parameters(self.model)

    def fit(self):
        self.trainer.fit(self.model)

    def eval(self) -> Dict[str, Any]:
        test_results = self.trainer.validate(self.model)
        test_result = test_results[0]
        loss = test_result.get('val_loss', 0.0)
        wer = test_result.get('val_wer', 0.0)
        return {"val_loss": loss, "val_wer": wer}


class FedAvgServer:
    def __init__(self, rounds: int, clients_per_round: int):
        self.rounds = rounds
        self.clients_per_round = clients_per_round
        self.global_model_parameters = None

    def run_simulation(self):
        global LOGGER
        print(f"--- INFO ---  Creating initial client")
        init_client = NemoFedClient(0)
        self.global_model_parameters = init_client.get_parameters()

        print(f"--- INFO ---  Started evaluating initial client")
        init_eval_vals = init_client.eval()
        print(f"--- INFO ---  Finished evaluating initial client")
        loss = init_eval_vals.get('val_loss', 0.0)
        wer = init_eval_vals.get('val_wer', 0.0)
        LOGGER.experiment.add_scalar("round_val_loss", scalar_value=loss, global_step=0)
        LOGGER.experiment.add_scalar("round_val_wer", scalar_value=wer, global_step=0)
        for r in range(1, self.rounds + 1):

            print(f"--- INFO ---  Starting round {r}")
            client_parameters = []
            for c in range(self.clients_per_round):
                print(f"--- INFO ---  Creating client {c} for round {r}")
                log_lr = False
                if c == 0:
                    # log learning rate only for the first client, all others should have equal learning rate anyways
                    log_lr = True
                client = NemoFedClient(r, log_lr)
                client.set_parameters(self.global_model_parameters)

                print(f"--- INFO ---  Starting fitting of client {c} for round {r}")
                client.fit()

                print(f"--- INFO ---  Finished fitting of client {c} for round {r}")
                model_params = client.get_parameters()
                client_parameters.append(model_params)

            print(f"--- INFO ---  Started averaging for round {r}")
            self.global_model_parameters = self.average_parameters(client_parameters)

            print(f"--- INFO ---  Creating averaged global model for round {r}")
            eval_client = NemoFedClient(r)
            eval_client.set_parameters(self.global_model_parameters)

            print(f"--- INFO ---  Starting evaluation of averaged global model for round {r}")
            eval_vals = eval_client.eval()
            print(f"--- INFO ---  Finished evaluation of averaged global model for round {r}")
            loss = eval_vals.get('val_loss', 0.0)
            wer = eval_vals.get('val_wer', 0.0)
            LOGGER.experiment.add_scalar("round_val_loss", scalar_value=loss, global_step=r)
            LOGGER.experiment.add_scalar("round_val_wer", scalar_value=wer, global_step=r)

            print(f"--- INFO ---  Finished round {r}")

    def average_parameters(self, clients_list: List[List[np.ndarray]]) -> List[np.ndarray]:
        num_layers = len(clients_list[0])
        averaged_weights = [None] * num_layers

        for i in range(num_layers):
            layer_stack = np.stack([client[i] for client in clients_list])
            mean_layer = np.mean(layer_stack, axis=0)
            averaged_weights[i] = mean_layer

        return averaged_weights

fed_server = FedAvgServer(ROUNDS, CLIENTS_PER_ROUND)
fed_server.run_simulation()
