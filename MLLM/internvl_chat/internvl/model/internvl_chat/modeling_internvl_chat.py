# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import warnings
import pdb
import math
from typing import Any, List, Optional, Tuple, Union

import torch.utils.checkpoint
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.phi3.modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModel,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    Qwen2ForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel
from .modeling_intern_pcmae import InternPointCloudModel

logger = logging.get_logger(__name__)


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = "pixel_values"
    _no_split_modules = [
        "InternVisionEncoderLayer",
        "LlamaDecoderLayer",
        "InternLM2DecoderLayer",
        "Phi3DecoderLayer",
        "Qwen2DecoderLayer",
        "TransformerEncoder",
    ]

    def __init__(
        self, config: InternVLChatConfig, vision_model=None, language_model=None
    ):
        super().__init__(config)

        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
        )
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version

        logger.info(f"num_image_token: {self.num_image_token}")
        logger.info(f"ps_version: {self.ps_version}")
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == "LlamaForCausalLM":
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == "InternLM2ForCausalLM":
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == "Phi3ForCausalLM":
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == "Qwen2ForCausalLM":
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(
                    f"{config.llm_config.architectures[0]} is not implemented."
                )

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size
            ),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

        self.use_point_cloud_model = config.point_cloud_config.use_point_cloud_model
        # if self.use_point_cloud_model:
        self.point_cloud_model = InternPointCloudModel(config.point_cloud_config)
        self.point_cloud_model.load_model_from_ckpt(
            config.point_cloud_config.checkpoint_path
        )
        self.mlp2 = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, llm_hidden_size // 4),
            nn.GELU(),
            # nn.LayerNorm(llm_hidden_size//2),
            nn.Linear(llm_hidden_size // 4, llm_hidden_size // 2),
            nn.GELU(),
            nn.Linear(llm_hidden_size // 2, llm_hidden_size),
        )
        # nn.init.kaiming_uniform_(self.mlp2[1].weight, nonlinearity='relu')
        # nn.init.constant_(self.mlp2[1].bias, 0)
        # nn.init.xavier_uniform_(self.mlp2[3].weight)
        # nn.init.constant_(self.mlp2[3].bias, 0)

        # if config.force_image_size != config.vision_config.image_size:
        #     self.vision_model.resize_pos_embeddings(
        #         old_size=config.vision_config.image_size,
        #         new_size=config.force_image_size,
        #         patch_size=config.vision_config.patch_size
        #     )

        self.img_context_token_id = None
        self.depth_context_token_id = None
        self.pc_context_token_id = None
        self.map_context_token_id = None
        self.odom_context_token_id = None
        self.neftune_alpha = None

        self.use_odometry_model = getattr(config, "use_odometry_model", True)
        self.odometry_num_freq = getattr(config, "odometry_num_freq", 6)
        self.odometry_input_dim = getattr(config, "odometry_input_dim", 6)
        self.odometry_token_repeats = self.num_image_token
        odometry_hidden_dim = getattr(
            config, "odometry_hidden_dim", llm_hidden_size // 2
        )
        odom_fourier_dim = self.odometry_input_dim * (1 + 2 * self.odometry_num_freq)
        self.mlp3 = nn.Sequential(
            nn.LayerNorm(odom_fourier_dim),
            nn.Linear(odom_fourier_dim, odometry_hidden_dim),
            nn.GELU(),
            nn.Linear(odometry_hidden_dim, llm_hidden_size),
        )

        if config.use_backbone_lora:
            self.wrap_backbone_lora(
                r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora
            )

        if config.use_llm_lora:
            self.wrap_llm_lora(
                r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora
            )

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=[
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.down_proj",
                "mlp.up_proj",
            ],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type="CAUSAL_LM",
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        point_cloud: torch.FloatTensor = None,
        odometry: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        point_cloud_flags: Optional[torch.LongTensor] = None,
        odometry_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        image_flags = image_flags.squeeze(-1)
        # pdb.set_trace()
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
        # print('input_embeds is' ,input_embeds)
        # print('input_embeds len is' ,len(input_embeds))
        # print('input_embeds shape is', input_embeds.shape)
        # pdb.set_trace()
        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        # print('input_embeds len is' ,len(input_embeds))
        # print('input_embeds shape is' ,input_embeds.shape)
        # pdb.set_trace()
        if self.use_point_cloud_model:
            point_cloud_flags = point_cloud_flags.reshape(-1)
            point_cloud_flags = point_cloud_flags.squeeze(-1)
            point_cloud_embeds = self.extract_pc_feature(point_cloud)
            point_cloud_embeds = point_cloud_embeds[point_cloud_flags == 1]

        odometry_embeds = None
        if (
            self.use_odometry_model
            and odometry is not None
            and odometry_flags is not None
        ):
            odometry_flags = odometry_flags.reshape(-1)
            odometry_flags = odometry_flags.squeeze(-1)
            odometry_embeds = self.extract_odom_feature(odometry)
            odometry_embeds = odometry_embeds[odometry_flags == 1]

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(
                f"dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}"
            )
        # pdb.set_trace()
        input_ids = input_ids.reshape(B * N)
        selected_img = input_ids == self.img_context_token_id
        selected_pc = input_ids == self.pc_context_token_id
        selected_odom = input_ids == self.odom_context_token_id
        # pdb.set_trace()
        # print('self.img_context_token_id is :', self.img_context_token_id)
        # print('self.pc_context_token_id is :', self.pc_context_token_id)
        # print('selected is :', selected_img)

        if self.use_point_cloud_model:
            try:
                input_embeds[selected_img] = input_embeds[
                    selected_img
                ] * 0.0 + vit_embeds.reshape(-1, C)
                input_embeds[selected_pc] = input_embeds[
                    selected_pc
                ] * 0.0 + point_cloud_embeds.reshape(-1, C)
                if odometry_embeds is not None:
                    input_embeds[selected_odom] = input_embeds[
                        selected_odom
                    ] * 0.0 + odometry_embeds.reshape(-1, C)
                ignore_flag = False
                # pdb.set_trace()
            except Exception as e:
                vit_embeds = vit_embeds.reshape(-1, C)
                print(
                    f"warning: {e}, input_embeds[selected].shape={input_embeds[selected_img].shape}, "
                    f"vit_embeds.shape={vit_embeds.shape}"
                )
                n_token = selected_img.sum()
                input_embeds[selected_img] = (
                    input_embeds[selected_img] * 0.0
                    + vit_embeds[:n_token]
                    + point_cloud_embeds[:n_token]
                )
                if odometry_embeds is not None:
                    input_embeds[selected_odom] = (
                        input_embeds[selected_odom] * 0.0
                        + odometry_embeds[: selected_odom.sum()]
                    )
                ignore_flag = True
        else:
            try:
                input_embeds[selected_img] = input_embeds[
                    selected_img
                ] * 0.0 + vit_embeds.reshape(-1, C)
                if odometry_embeds is not None:
                    input_embeds[selected_odom] = input_embeds[
                        selected_odom
                    ] * 0.0 + odometry_embeds.reshape(-1, C)
                ignore_flag = False
            except Exception as e:
                vit_embeds = vit_embeds.reshape(-1, C)
                print(
                    f"warning: {e}, input_embeds[selected].shape={input_embeds[selected_img].shape}, "
                    f"vit_embeds.shape={vit_embeds.shape}"
                )
                n_token = selected_img.sum()
                input_embeds[selected_img] = (
                    input_embeds[selected_img] * 0.0 + vit_embeds[:n_token]
                )
                if odometry_embeds is not None:
                    input_embeds[selected_odom] = (
                        input_embeds[selected_odom] * 0.0
                        + odometry_embeds[: selected_odom.sum()]
                    )
                ignore_flag = True

        input_embeds = input_embeds.reshape(B, N, C)
        # pdb.set_trace()
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        print("ignore_flag is :", ignore_flag)
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self.ps_version == "v1":
            warnings.warn(
                "In ps_version 'v1', the height and width have not been swapped back, "
                "which results in a transposed image."
            )
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def noised_embed(self, vit_embeds, noise_alpha=5):
        dims = torch.tensor(vit_embeds.size(1) * vit_embeds.size(2))
        mag_norm = noise_alpha / torch.sqrt(dims)
        noise = torch.zeros_like(vit_embeds).uniform_(-mag_norm, mag_norm)
        return vit_embeds + noise

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        if self.training and self.neftune_alpha is not None:
            vit_embeds = self.noised_embed(vit_embeds, self.neftune_alpha)

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

        vit_embeds = self.mlp1(vit_embeds)

        return vit_embeds

    def extract_pc_feature(self, point_cloud):
        point_cloud_embeds = self.point_cloud_model(point_cloud)
        # pdb.set_trace()
        point_cloud_embeds = self.mlp2(point_cloud_embeds)  # .to(pixel_values.device)
        # pdb.set_trace()
        return point_cloud_embeds

    def fourier_encode_odometry(self, odometry):
        base = [odometry]
        for idx in range(self.odometry_num_freq):
            freq = (2**idx) * math.pi
            base.append(torch.sin(freq * odometry))
            base.append(torch.cos(freq * odometry))
        return torch.cat(base, dim=-1)

    def extract_odom_feature(self, odometry):
        B, R, D = odometry.shape
        odometry = odometry.reshape(B * R, D)
        odometry_embeds = self.fourier_encode_odometry(odometry)
        odometry_embeds = self.mlp3(odometry_embeds)
        odometry_embeds = odometry_embeds.unsqueeze(1).repeat(
            1, self.odometry_token_repeats, 1
        )
        return odometry_embeds

    def batch_chat(
        self,
        tokenizer,
        pixel_values,
        image_counts,
        questions,
        generation_config,
        history=None,
        return_history=False,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
    ):
        if history is not None or return_history:
            print("Now multi-turn chat is not supported in batch_chat.")
            raise NotImplementedError
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        from .conversation import get_conv_template

        queries = []
        image_bs = pixel_values.shape[0]
        # print(f'dynamic ViT batch size: {image_bs}, image_counts: {image_counts}')
        for idx, image_count in enumerate(image_counts):
            image_token = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * image_count
                + IMG_END_TOKEN
            )
            question = image_token + "\n" + questions[idx]
            template = get_conv_template(self.template)
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()
            queries.append(query)
        tokenizer.padding_side = "left"
        model_inputs = tokenizer(queries, return_tensors="pt", padding=True)
        input_ids = model_inputs["input_ids"].cuda()
        attention_mask = model_inputs["attention_mask"].cuda()
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config["eos_token_id"] = eos_token_id

        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep)[0].strip() for response in responses]
        return responses

    def chat(
        self,
        tokenizer,
        pixel_values,
        question,
        generation_config,
        point_cloud=None,
        map_point_cloud=None,
        depth_values=None,
        odometry=None,
        history=None,
        return_history=False,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        DEPTH_START_TOKEN="<depth>",
        DEPTH_END_TOKEN="</depth>",
        DEPTH_CONTEXT_TOKEN="<DEPTH_CONTEXT>",
        PC_START_TOKEN="<pointcloud>",
        PC_END_TOKEN="</pointcloud>",
        PC_CONTEXT_TOKEN="<PC_CONTEXT>",
        MAP_START_TOKEN="<map>",
        MAP_END_TOKEN="</map>",
        MAP_CONTEXT_TOKEN="<MAP_CONTEXT>",
        ODOM_START_TOKEN="<odometry>",
        ODOM_END_TOKEN="</odometry>",
        ODOM_CONTEXT_TOKEN="<ODOM_CONTEXT>",
    ):

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id
        # print('img_context_token_id is ',img_context_token_id)
        pc_context_token_id = tokenizer.convert_tokens_to_ids(PC_CONTEXT_TOKEN)
        self.pc_context_token_id = pc_context_token_id
        map_context_token_id = tokenizer.convert_tokens_to_ids(MAP_CONTEXT_TOKEN)
        self.map_context_token_id = map_context_token_id
        depth_context_token_id = tokenizer.convert_tokens_to_ids(DEPTH_CONTEXT_TOKEN)
        self.depth_context_token_id = depth_context_token_id
        odom_context_token_id = tokenizer.convert_tokens_to_ids(ODOM_CONTEXT_TOKEN)
        self.odom_context_token_id = odom_context_token_id
        # print('pc_context_token_id is ',pc_context_token_id)
        from internvl.conversation import get_conv_template

        template = get_conv_template(self.template)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        image_bs = pixel_values.shape[0]
        image_tokens = ""
        depth_tokens = ""
        pc_tokens = ""
        map_tokens = ""
        odom_tokens = ""
        print(f"dynamic ViT batch size: {image_bs}")
        if history is None:
            history = []
            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * image_bs
                + IMG_END_TOKEN
            )
            if depth_values is not None:
                depth_tokens = (
                    DEPTH_START_TOKEN
                    + DEPTH_CONTEXT_TOKEN * self.num_image_token * image_bs
                    + DEPTH_END_TOKEN
                )
            pc_tokens = (
                PC_START_TOKEN
                + PC_CONTEXT_TOKEN * self.num_image_token * image_bs
                + PC_END_TOKEN
            )
            if map_point_cloud is not None:
                map_tokens = (
                    MAP_START_TOKEN
                    + MAP_CONTEXT_TOKEN * self.num_image_token * image_bs
                    + MAP_END_TOKEN
                )

            all_prefix_tokens = [image_tokens]
            if depth_tokens:
                all_prefix_tokens.append(depth_tokens)
            all_prefix_tokens.append(pc_tokens)
            if map_tokens:
                all_prefix_tokens.append(map_tokens)
            if odometry is not None:
                odom_tokens = (
                    ODOM_START_TOKEN
                    + ODOM_CONTEXT_TOKEN * self.num_image_token * image_bs
                    + ODOM_END_TOKEN
                )
                all_prefix_tokens.append(odom_tokens)

            question = "\n".join(all_prefix_tokens + [question])
            # print('question is :', question)
        else:
            for old_question, old_answer in history:
                template.append_message(template.roles[0], old_question)
                template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        # print('query is :', query)
        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].cuda()
        # print('input_ids is :', input_ids)
        # print('input_ids len is  :', len(input_ids))
        attention_mask = model_inputs["attention_mask"].cuda()
        generation_config["eos_token_id"] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            point_cloud=point_cloud,
            map_point_cloud=map_point_cloud,
            depth_values=depth_values,
            odometry=odometry,
            **generation_config,
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[
            0
        ]
        # print('respone 0 is:', response)
        response = response.split(template.sep)[0].strip()
        # print('respone 1 is:', response)
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(image_tokens, "<image>")
            if depth_values is not None:
                query_to_print = query_to_print.replace(depth_tokens, "<depth>")
            query_to_print = query_to_print.replace(pc_tokens, "<pointcloud>")
            if map_point_cloud is not None:
                query_to_print = query_to_print.replace(map_tokens, "<map>")
            if odometry is not None:
                query_to_print = query_to_print.replace(odom_tokens, "<odometry>")
            print(query_to_print, response)
            return response
        return response

    def multi_image_chat(
        self,
        tokenizer,
        pixel_values,
        image_counts,
        question,
        generation_config,
        history=None,
        return_history=False,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
    ):

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        from internvl.conversation import get_conv_template

        template = get_conv_template(self.template)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        if history is None:
            history = []
            image_tokens = ""
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}, image_counts: {image_counts}")
            for idx, image_count in enumerate(image_counts):
                image_tokens += (
                    f"<image {idx + 1}> (图{idx + 1}):"
                    + IMG_START_TOKEN
                    + IMG_CONTEXT_TOKEN * self.num_image_token * image_count
                    + IMG_END_TOKEN
                )
            question = image_tokens + "\n" + question
        else:
            for old_question, old_answer in history:
                template.append_message(template.roles[0], old_question)
                template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].cuda()
        attention_mask = model_inputs["attention_mask"].cuda()
        generation_config["eos_token_id"] = eos_token_id

        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[
            0
        ]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(image_tokens, "<image>")
            print(query_to_print, response)
            return response
        return response

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        point_cloud: Optional[torch.FloatTensor] = None,
        map_point_cloud: Optional[torch.FloatTensor] = None,
        depth_values: Optional[torch.FloatTensor] = None,
        odometry: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if (
            pixel_values is not None
            or depth_values is not None
            or point_cloud is not None
            or map_point_cloud is not None
            or odometry is not None
        ):
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = (
                    self.extract_feature(pixel_values)
                    if pixel_values is not None
                    else None
                )

            depth_embeds = (
                self.extract_feature(depth_values) if depth_values is not None else None
            )

            point_cloud_embeds = (
                self.extract_pc_feature(point_cloud)
                if point_cloud is not None
                else None
            )
            map_embeds = (
                self.extract_pc_feature(map_point_cloud)
                if map_point_cloud is not None
                else None
            )
            odometry_embeds = (
                self.extract_odom_feature(odometry) if odometry is not None else None
            )

            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            if vit_embeds is not None:
                selected_img = input_ids == self.img_context_token_id
                assert selected_img.sum() != 0
                input_embeds[selected_img] = vit_embeds.reshape(-1, C).to(
                    input_embeds.device
                )

            if depth_embeds is not None:
                selected_depth = input_ids == self.depth_context_token_id
                assert selected_depth.sum() != 0
                input_embeds[selected_depth] = depth_embeds.reshape(-1, C).to(
                    input_embeds.device
                )

            if point_cloud_embeds is not None:
                selected_pc = input_ids == self.pc_context_token_id
                assert selected_pc.sum() != 0
                input_embeds[selected_pc] = point_cloud_embeds.reshape(-1, C).to(
                    input_embeds.device
                )

            if map_embeds is not None:
                selected_map = input_ids == self.map_context_token_id
                assert selected_map.sum() != 0
                input_embeds[selected_map] = map_embeds.reshape(-1, C).to(
                    input_embeds.device
                )

            if odometry_embeds is not None:
                selected_odom = input_ids == self.odom_context_token_id
                assert selected_odom.sum() != 0
                input_embeds[selected_odom] = odometry_embeds.reshape(-1, C).to(
                    input_embeds.device
                )

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs


# AutoModel.register(InternVLChatConfig,InternVLChatModel)
