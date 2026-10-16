# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import os
from typing import Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig

logger = logging.get_logger(__name__)


class InternPointCloudConfig(PretrainedConfig):
    # model_type = 'intern_vit_6b'

    def __init__(
        self,
        group_size=16,
        num_group=256,
        encoder_dims=384,
        trans_dim=384,
        depth=12,
        drop_path_rate=0.1,
        num_heads=6,
        checkpoint_path="",
        use_point_cloud_model=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.group_size = group_size
        self.num_group = num_group
        self.encoder_dims = encoder_dims
        self.trans_dim = trans_dim
        self.depth = depth
        self.drop_path_rate = drop_path_rate
        self.num_heads = num_heads
        self.checkpoint_path = checkpoint_path
        self.use_point_cloud_model = use_point_cloud_model

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )

        if "point_cloud_config" in config_dict:
            config_dict = config_dict["point_cloud_config"]

        if (
            "model_type" in config_dict
            and hasattr(cls, "model_type")
            and config_dict["model_type"] != cls.model_type
        ):
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


# AutoConfig.register("configuration_intern_pcmae.InternPointCloudConfig", InternPointCloudConfig)
