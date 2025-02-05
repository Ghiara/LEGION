"""Component to encode the task."""

import json

import torch
import torch.nn as nn

from mtrl.agent import utils as agent_utils
from mtrl.agent.components import base as base_component
from mtrl.utils.types import ConfigType, TensorType


############################################
######### Metadata Context Encoder #########
############################################

class TaskEncoder(base_component.Component):
    def __init__(
        self, # cfg refer multitask task encoder cfg
        pretrained_embedding_cfg: ConfigType, # should use, path to load from, ordered task list
        num_embeddings: int, # agent.multitask.num_envs -> envs.num_envs MT10 -> 10
        embedding_dim: int, # 50
        hidden_dim: int, # 50
        num_layers: int, # 2
        output_dim: int, # 50
    ):
        """
        Context Encoder, encode the context metadata in to a vector z_context
        Encode the task into a vector.

        Args:
            pretrained_embedding_cfg (ConfigType): config for using pretrained
                embeddings.
            
            num_embeddings (int): number of elements in the embedding table. This is
                used if pretrained embedding is not used.
            embedding_dim (int): dimension for the embedding. This is
                used if pretrained embedding is not used.
            
            hidden_dim (int): dimension of the hidden layer of the trunk.
            num_layers (int): number of layers in the trunk.
            output_dim (int): output dimension of the task encoder.
            
            TaskEncoder(
                (embedding): Sequential(
                    (0): Embedding(10, 768)
                    (1): ReLU()
                    (2): Sequential(
                        (0): Linear(in_features=768, out_features=100, bias=True)
                        (1): ReLU()
                        (2): Linear(in_features=100, out_features=50, bias=True)
                        (3): ReLU()
                    )
                )
                (trunk): Sequential(
                    (0): Linear(in_features=50, out_features=50, bias=True)
                    (1): ReLU()
                    (2): Linear(in_features=50, out_features=50, bias=True)
                    (3): ReLU()
                    (4): Linear(in_features=50, out_features=50, bias=True)
                )
            )

        """
        super().__init__()
        ##############################################################################
        if pretrained_embedding_cfg.should_use:
            ######################################
            ######### Pretrained NLP model #######
            ######################################
            with open(pretrained_embedding_cfg.path_to_load_from) as f:
                metadata = json.load(f) # dict with key='task name, e.g. reach-v1', value = list with 768 length
            # ordered_task_list from experiment, build_envs(), metadata
            ordered_task_list = pretrained_embedding_cfg.ordered_task_list # provide task name
            
            # check and replace the v2 to v1
            if ordered_task_list[0].split('-')[-1] == 'v2':
                new_ordered_task_list = []
                for idx, name in enumerate(ordered_task_list):
                    # replace all v2 to v1 to fit the metadata name
                    new_name = name.replace(name[-1], '1')
                    new_ordered_task_list.append(new_name)
            else:
                new_ordered_task_list = ordered_task_list

            pretrained_embedding = torch.Tensor(
                [metadata[task] for task in new_ordered_task_list]
            )
            assert num_embeddings == pretrained_embedding.shape[0] # num embedding == num_envs
            pretrained_embedding_dim = pretrained_embedding.shape[1] # Input of pretrained NLP model = 768 in mt10
            
            # TODO modify to suitable for CRL input
            self.pretrained_embedding = nn.Embedding.from_pretrained(
                embeddings=pretrained_embedding,
                freeze=True,
                )

            # projection_layer = nn.Sequential(
            #     nn.Linear(
            #         in_features=pretrained_embedding_dim, out_features=2 * embedding_dim # (768, 100)
            #     ),
            #     nn.ReLU(),
            #     # nn.Linear(in_features=2 * embedding_dim, out_features=embedding_dim),
            #     nn.Linear(in_features=2 * embedding_dim, out_features=output_dim), # (100, output)
            #     nn.ReLU(),
            #     )
            # projection_layer.apply(agent_utils.weight_init)
            
            # self.embedding = nn.Sequential(  # type: ignore [call-overload]
            #     pretrained_embedding,
            #     nn.ReLU(),
            #     projection_layer,
            # )
        ##########################################################################################
        else:
            self.embedding = nn.Sequential(
                nn.Embedding(
                    num_embeddings=num_embeddings, embedding_dim=embedding_dim
                ),
                nn.ReLU(),
            )
            self.embedding.apply(agent_utils.weight_init)
        
        ######################################
        ######### context encoder MLP ########
        ######################################
        # self.trunk = agent_utils.build_mlp(
        #     input_dim=embedding_dim,
        #     hidden_dim=hidden_dim,
        #     output_dim=output_dim,
        #     num_layers=num_layers,
        # )
        # self.trunk.apply(agent_utils.weight_init)

    def forward(self, env_index: TensorType) -> TensorType:
        # return self.trunk(self.embedding(env_index))
        # return self.embedding(env_index)
        return self.pretrained_embedding(env_index)





class TaskEncoder_CARE(base_component.Component):
    def __init__(
        self,
        pretrained_embedding_cfg: ConfigType,
        num_embeddings: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
    ):
        """Encode the task into a vector.

        Args:
            pretrained_embedding_cfg (ConfigType): config for using pretrained
                embeddings.
            num_embeddings (int): number of elements in the embedding table. This is
                used if pretrained embedding is not used.
            embedding_dim (int): dimension for the embedding. This is
                used if pretrained embedding is not used.
            hidden_dim (int): dimension of the hidden layer of the trunk.
            num_layers (int): number of layers in the trunk.
            output_dim (int): output dimension of the task encoder.
        """
        super().__init__()
        if pretrained_embedding_cfg.should_use:
            with open(pretrained_embedding_cfg.path_to_load_from) as f:
                metadata = json.load(f)
            ordered_task_list = pretrained_embedding_cfg.ordered_task_list

            # check and replace the v2 to v1
            if ordered_task_list[0].split('-')[-1] == 'v2':
                new_ordered_task_list = []
                for idx, name in enumerate(ordered_task_list):
                    # replace all v2 to v1 to fit the metadata name
                    new_name = name.replace(name[-1], '1')
                    new_ordered_task_list.append(new_name)
            else:
                new_ordered_task_list = ordered_task_list

            pretrained_embedding = torch.Tensor(
                [metadata[task] for task in ordered_task_list]
            )
            assert num_embeddings == pretrained_embedding.shape[0]
            pretrained_embedding_dim = pretrained_embedding.shape[1]
            pretrained_embedding = nn.Embedding.from_pretrained(
                embeddings=pretrained_embedding,
                freeze=True,
            )
            projection_layer = nn.Sequential(
                nn.Linear(
                    in_features=pretrained_embedding_dim, out_features=2 * embedding_dim
                ),
                nn.ReLU(),
                nn.Linear(in_features=2 * embedding_dim, out_features=embedding_dim),
                nn.ReLU(),
            )
            projection_layer.apply(agent_utils.weight_init)
            self.embedding = nn.Sequential(  # type: ignore [call-overload]
                pretrained_embedding,
                nn.ReLU(),
                projection_layer,
            )

        else:
            self.embedding = nn.Sequential(
                nn.Embedding(
                    num_embeddings=num_embeddings, embedding_dim=embedding_dim
                ),
                nn.ReLU(),
            )
            self.embedding.apply(agent_utils.weight_init)
        self.trunk = agent_utils.build_mlp(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
        )
        self.trunk.apply(agent_utils.weight_init)

    def forward(self, env_index: TensorType) -> TensorType:
        return self.trunk(self.embedding(env_index))