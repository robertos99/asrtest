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
from omegaconf import open_dict, DictConfig, OmegaConf
import pytorch_lightning as ptl
import pickle
from pytorch_lightning import Callback
import hydra
import logging
import sys
import random
import uuid


LOGGER = None # later set in main
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TOTAL_EPOCH = 0
TOTAL_STEP = 0

EPOCH_PER_ROUND = 5
DATA_PER_SPEAKER = 10
SPEAKER_PER_CLIENT = 8
ROUNDS = 10
CLIENTS_PER_ROUND = 10

class LogLearningRate(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global LOGGER, TOTAL_STEP

        lr = trainer.optimizers[0].param_groups[0]['lr']

        LOGGER.experiment.add_scalar("lr", scalar_value=lr, global_step=TOTAL_STEP)

        TOTAL_STEP += 1


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


def set_config(net, train_manifest_path, test_manifest_path, fed_config):
    cfg = net.cfg
    with (open_dict(cfg)):
        cfg.train_ds.manifest_filepath = train_manifest_path
        # cfg.train_ds.batch_size = 1
        cfg.train_ds.batch_size = fed_config.client.training.batch_size
        cfg.train_ds.num_workers = fed_config.client.training.num_workers
        cfg.train_ds.is_tarred = False
        #   cfg.train_ds.pin_memory=True \
        # cfg.test_ds.manifest_filepath = test_manifest_path
        # cfg.test_ds.batch_size = cfg.client.validation.batch_size
        # cfg.test_ds.num_workers = cfg.client.training.num_workers

        # Validation dataset  (Use test dataset as validation, since we train using train + dev)
        print(fed_config)
        cfg.validation_ds.manifest_filepath = test_manifest_path
        cfg.validation_ds.batch_size = fed_config.client.validation.batch_size
        cfg.validation_ds.num_workers = fed_config.client.validation.num_workers
        cfg.train_ds.is_tarred = False

        cfg.spec_augment.freq_masks = fed_config.client.augment.freq_mask
        cfg.spec_augment.time_masks = fed_config.client.augment.time_mask

        # optim defined elsewhere. we need to reset the nemos config to null effectively
        cfg.optim = {}

    net.cfg = net._cfg
    net.setup_training_data(cfg.train_ds)
    net.setup_validation_data(cfg.validation_ds)
    net.setup_optimization(cfg.optim)
    net.spec_augmentation = net.from_config_dict(cfg.spec_augment)



class FedSchedEncEecCtcBpe(nemo_asr.models.EncDecCTCModelBPE):

    def __init__(self, cfg: DictConfig, trainer=None):
        super().__init__(cfg, trainer)
        self.last_lr_step = None
        self.non_nemo_cfg = None

    def set_last_lr_step(self, last_lr_step):
        self.last_lr_step = last_lr_step

    def set_non_nemo_cfg(self, cfg: DictConfig):
        self.non_nemo_cfg = cfg

    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        logging.info("called FedSchedEncEecCtcBpe configure_optimizers")
        lr = self.non_nemo_cfg.client.model.optim.initial_lr
        betas = self.non_nemo_cfg.client.model.optim.betas
        weight_decay = self.non_nemo_cfg.client.model.optim.weight_decay
        optimizer = AdamW(self.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = lr
        if self.non_nemo_cfg.client.model.optim.sched.type == "sched":
            data_per_speaker = self.non_nemo_cfg.client.training.data_per_speaker
            speaker_per_client = self.non_nemo_cfg.client.training.speaker_per_client
            train_samples = data_per_speaker * speaker_per_client
            steps_per_epoch = train_samples // self.non_nemo_cfg.client.training.batch_size
            max_steps = steps_per_epoch * self.non_nemo_cfg.client.training.epoch_per_round * self.non_nemo_cfg.federated_strategy.rounds


            min_lr = self.non_nemo_cfg.client.model.optim.sched.min_lr
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max_steps,
                eta_min=min_lr,
                last_epoch=self.last_lr_step
            )
            return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
        else:
            # no scheduler
            return optimizer

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

    # Create a unique filename
    clientuuid = uuid.uuid4()
    merged_manifest_path = os.path.join(tmp_dir, f'merged_manifest_{clientuuid}.json')

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

class FedDataLoader(ABC):
    @abstractmethod
    def get_next_manifest(self, clientid):
        pass

class NonIIDDataLoader(FedDataLoader):
    def __init__(self, num_total_clients, speaker_per_client):
        self.fed_manifests_folder_path = "timit-dataset/federated-manifests"
        self.speaker_manifest_paths = read_speaker_manifests(self.fed_manifests_folder_path)
        self.tmp_dir = 'tmp-manifest'
        self.client_manifest_path = {}
        for i in range(0, num_total_clients):
            logging.info(f"generating manifest for client {i}")
            train_manifest_paths = []
            for _ in range(speaker_per_client):
                train_manifest_paths.append(self.speaker_manifest_paths.pop())
            train_manifest_path = merge_manifests(train_manifest_paths, self.tmp_dir)
            self.client_manifest_path[i] = train_manifest_path
        print(f"clients manifests {self.client_manifest_path}")

    def get_next_manifest(self, clientid):
        return self.client_manifest_path[clientid]

class IIDDataLoader(FedDataLoader):
    def __init__(self, num_total_clients, speaker_per_client):
        self.fed_manifests_folder_path = "timit-dataset/federated-manifests"
        self.speaker_manifest_paths = read_speaker_manifests(self.fed_manifests_folder_path)
        self.tmp_dir = 'tmp-iid-manifest'
        all_merged_path = merge_manifests(list(self.speaker_manifest_paths), self.tmp_dir)
        all_transcriptions = []
        with open(all_merged_path, 'r') as f:
            for line in f:
                all_transcriptions.append(line)
        self.client_manifest_path = {}
        for i in range(0, num_total_clients):
            logging.info(f"generating manifest for client {i}")
            # Create a unique filename
            clientuuid = uuid.uuid4()
            train_manifest_path = os.path.join(self.tmp_dir, f'merged_manifest_{clientuuid}.json')
            iid_client_transcriptions = []
            # 10 samples per speaker
            for _ in range(speaker_per_client*10):
                random_transcription = random.choice(all_transcriptions)
                all_transcriptions.remove(random_transcription)
                iid_client_transcriptions.append(random_transcription)
            with open(train_manifest_path, 'w') as train_file:
                for nemo_transcription in iid_client_transcriptions:
                    train_file.write(nemo_transcription)
            self.client_manifest_path[i] = train_manifest_path
        print(f"clients manifests {self.client_manifest_path}")

    def get_next_manifest(self, clientid):
        return self.client_manifest_path[clientid]

TEST_MANIFEST_PATH = "timit-dataset/test-manifest.clean.json"


class NemoFedClient(FedClient):
    # round starts from 1 since 0 is before we finetune
    # clientid starts from 0 like everything else
    # dataloader can be None for validaton clients so we dont pop data
    def __init__(self, round: int, clientid: int, cfg: DictConfig, dataloader: FedDataLoader | None, log_lr=False):
        global TEST_MANIFEST_PATH

        # this is a dummy value which will be overridden by all clients that are used for training. The None is only for validation
        train_manifest_path = "timit-dataset/train-manifest.clean.json"
        if dataloader is not None:
            # if dataloader is not none its a trainig client
            train_manifest_path = dataloader.get_next_manifest(clientid)

        self.model = FedSchedEncEecCtcBpe.from_pretrained(model_name="stt_en_squeezeformer_ctc_xsmall_ls")
        train_manifest_path = train_manifest_path
        set_config(self.model, train_manifest_path, TEST_MANIFEST_PATH, cfg)
        self.model.set_non_nemo_cfg(cfg)

        epoch_per_round = cfg.client.training.epoch_per_round
        self.model.set_last_lr_step(TOTAL_STEP)

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

# receive one random client. if a client is returned it is removed from the available clients.
# once ALL clients are removed all clients become available again.
# this is to guaranntee each clients data is used the exact same amount of time as any other.
class ClientSelector:
    def __init__(self, num_total_clients: int):
        self.num_total_clients = num_total_clients
        self.clients = list(range(num_total_clients))
        self.available_clients = self.clients.copy()

    def get_next_client(self) -> int:
        if not self.available_clients:
            self.available_clients = self.clients.copy()  # Reset the list when all clients have been selected

        client_id = random.choice(self.available_clients)
        self.available_clients.remove(client_id)
        return client_id

class FedAvgServer:
    def __init__(self, cfg: DictConfig, output_dir: str):
        self.output_dir = output_dir
        dataloader = None
        if cfg.federated_strategy.data == "noniid":
            dataloader = NonIIDDataLoader(cfg.client.training.num_total_clients, cfg.client.training.speaker_per_client)
        if cfg.federated_strategy.data == "iid":
            dataloader = IIDDataLoader(cfg.client.training.num_total_clients, cfg.client.training.speaker_per_client)
        self.dataloader = dataloader
        self.rounds = cfg.federated_strategy.rounds
        self.validate_every_n_rounds = cfg.federated_strategy.validate_every_n_rounds
        self.clients_per_round = cfg.federated_strategy.clients_per_round
        self.global_model_parameters = None
        self.cfg = cfg
        self.client_selector = ClientSelector(self.cfg.client.training.num_total_clients)

    def run_simulation(self):
        global LOGGER
        logging.info("Creating initial client")
        init_client = NemoFedClient(0, 0, self.cfg, None)
        self.global_model_parameters = init_client.get_parameters()

        logging.info("Started evaluating initial client")
        init_eval_vals = init_client.eval()
        logging.info("Finished evaluating initial client")
        loss = init_eval_vals.get('val_loss', 0.0)
        wer = init_eval_vals.get('val_wer', 0.0)
        LOGGER.experiment.add_scalar("round_val_loss", scalar_value=loss, global_step=0)
        LOGGER.experiment.add_scalar("round_val_wer", scalar_value=wer, global_step=0)
        for r in range(1, self.rounds + 1):
            logging.info(f"Starting round {r}")

            # Ensure the zeros are initialized as float32
            cumulative_parameters = [np.zeros_like(param, dtype=np.float32) for param in self.global_model_parameters]

            for c in range(self.clients_per_round):
                clientid = self.client_selector.get_next_client()
                logging.info(f"Creating client {c} with clientid {clientid} for round {r}")
                # only one client per round logs learning rate
                client = NemoFedClient(r, clientid, self.cfg, self.dataloader, log_lr=(c == 0))
                client.set_parameters(self.global_model_parameters)

                logging.info(f"Fitting client {c} for round {r}")
                client.fit()
                client_params = client.get_parameters()

                # Sum up the global model parameters
                for i, param in enumerate(client_params):
                    cumulative_parameters[i] += param
                del client  # Free the client object after use

            # Divide summed parameters by the number of clients
            client_count = np.float32(self.clients_per_round)
            self.global_model_parameters = [param / client_count for param in cumulative_parameters]
            if r %  self.validate_every_n_rounds == 0:
                eval_client = NemoFedClient(r, 0, self.cfg, None)
                eval_client.set_parameters(self.global_model_parameters)

                # Save the eval client weights after each round
                checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"model_weight_round_{r}.pth")
                torch.save(self.global_model_parameters, checkpoint_path)

                logging.info(f"Saved checkpoint for round {r} at {checkpoint_path}")

                logging.info("Starting evaluation of averaged global model for round {r}")
                eval_vals = eval_client.eval()
                logging.info("Finished evaluation of averaged global model for round {r}")
                loss = eval_vals.get('val_loss', 0.0)
                wer = eval_vals.get('val_wer', 0.0)
                LOGGER.experiment.add_scalar("round_val_loss", scalar_value=loss, global_step=r)
                LOGGER.experiment.add_scalar("round_val_wer", scalar_value=wer, global_step=r)

            logging.info("Finished round {r}")

    def average_parameters(self, clients_list: List[List[np.ndarray]]) -> List[np.ndarray]:
        num_layers = len(clients_list[0])
        averaged_weights = [None] * num_layers

        for i in range(num_layers):
            layer_stack = np.stack([client[i] for client in clients_list])
            mean_layer = np.mean(layer_stack, axis=0)
            averaged_weights[i] = mean_layer

        return averaged_weights


@hydra.main(version_base=None, config_path="fedconfig", config_name="config")
def main(cfg: DictConfig) -> None:
    global LOGGER

    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logging.info(f"Output directory  : {hydra_output_dir}")
    LOGGER = TensorBoardLogger(hydra_output_dir, name="federated_learning")

    logging.info(f"Current Configuration:\n{OmegaConf.to_yaml(cfg)}")
    fed_server = FedAvgServer(cfg, output_dir=hydra_output_dir)
    fed_server.run_simulation()


if __name__ == "__main__":
    main()
