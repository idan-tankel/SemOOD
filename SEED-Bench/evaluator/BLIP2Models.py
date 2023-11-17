from PIL import Image
import wandb
import os
import torch
import yaml
import subprocess
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn
from lavis.models.blip_models import blip_retrieval

# from .blip_utils.blip_retrieval import blip_retrieval
# from .blip_utils.utils import MetricLogger
from lavis.models import load_model_and_preprocess
from transformers import Blip2ForConditionalGeneration, Blip2Processor
image_dir = "/net/mraid11/export/data/idanta/SEED/SEED-Bench-image"
# All of the below URLs are taken from, and most of the implementation are heavily inspired from the wonderful https://github.com/salesforce/BLIP repo.

download_urls = {
    "blip-flickr-base": {
        "model_url": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_flickr.pth",
        "config_url": "https://raw.githubusercontent.com/salesforce/BLIP/0480d94d5725a3d1aac66f21e6bf138ac17d323d/configs/retrieval_flickr.yaml",
        "bert_config_url": "https://raw.githubusercontent.com/salesforce/BLIP/0480d94d5725a3d1aac66f21e6bf138ac17d323d/configs/med_config.json"
    },

    "blip-coco-base": {
        "model_url": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth",
        "config_url": "https://github.com/salesforce/BLIP/raw/0480d94d5725a3d1aac66f21e6bf138ac17d323d/configs/retrieval_coco.yaml",
        "bert_config_url": "https://raw.githubusercontent.com/salesforce/BLIP/0480d94d5725a3d1aac66f21e6bf138ac17d323d/configs/med_config.json"
    },
}


class BLIPModelWrapper:
    def _init_(self, root_dir, device, variant="blip-flickr-base"):
        self.variant = variant
        self.correct_count = 0
        self.root_dir = root_dir
        self.config_path = os.path.join(root_dir, f"{self.variant}-config")
        self.model_path = os.path.join(root_dir, f"{self.variant}.pth")
        self.bert_config_path = os.path.join(root_dir, "configs", f"{self.variant}_med_config.json")
        if not (os.path.exists(self.config_path) and os.path.exists(self.model_path) and os.path.exists(self.bert_config_path)):
            self.download()

        config = yaml.load(open(self.config_path, 'r'), Loader=yaml.Loader)
        self.config = config
        self.config['k_test'] = 128
        config['med_config'] = self.bert_config_path
        model = blip_retrieval(pretrained=self.model_path, image_size=config['image_size'], vit=config['vit'],
                               vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                               queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'],
                               med_config=config['med_config'])
        self.model = model.to(device)
        self.model = self.model.eval()
        self.device = device

    def download(self):
        print(f"Downloading BLIP model to {self.root_dir}...")
        model_url = download_urls[self.variant]["model_url"]
        config_url = download_urls[self.variant]["config_url"]
        bert_config_url = download_urls[self.variant]["bert_config_url"]
        os.makedirs(os.path.join(self.root_dir, "configs"), exist_ok=True)
        subprocess.call(["wget", "-c", model_url, "-O", self.model_path])
        subprocess.call(["wget", "-c", config_url, "-O", self.config_path])
        subprocess.call(["wget", "-c", bert_config_url, "-O", self.bert_config_path])

    @torch.no_grad()
    def get_text_embeddings(self, texts, text_batch_size=256):
        num_text = len(texts)
        text_bs = 256
        text_ids = []
        text_embeds = []
        text_atts = []
        for i in range(0, num_text, text_bs):
            text = texts[i: min(num_text, i + text_bs)]
            text_input = self.model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
            text_output = self.model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
            text_embed = F.normalize(self.model.text_proj(text_output.last_hidden_state[:, 0, :]))
            text_embeds.append(text_embed)
            text_ids.append(text_input.input_ids)
            text_atts.append(text_input.attention_mask)

        text_embeds = torch.cat(text_embeds, dim=0)
        text_ids = torch.cat(text_ids, dim=0)
        text_atts = torch.cat(text_atts, dim=0)
        text_ids[:, 0] = self.model.tokenizer.enc_token_id
        return text_embeds, text_ids, text_atts

    @torch.no_grad()
    def get_image_embeddings(self, image_loader):
        image_feats = []
        image_embeds = []
        for batch in tqdm(image_loader):
            image = batch["image"]
            image = image.to(self.device)
            image_feat = self.model.visual_encoder(image)
            image_embed = self.model.vision_proj(image_feat[:, 0, :])
            image_embed = F.normalize(image_embed, dim=-1)

            image_feats.append(image_feat.cpu())
            image_embeds.append(image_embed)

        image_feats = torch.cat(image_feats, dim=0)
        image_embeds = torch.cat(image_embeds, dim=0)
        return image_feats, image_embeds

    @torch.no_grad()
    def get_retrieval_scores_dataset(self, loader):
        texts = loader.dataset.text
        metric_logger = MetricLogger(delimiter="  ")

        text_embeds, text_ids, text_atts = self.get_text_embeddings(texts)
        image_feats, image_embeds = self.get_image_embeddings(loader)
        sims_matrix = image_embeds @ text_embeds.t()
        score_matrix_i2t = torch.full((image_embeds.shape[0], len(texts)), -100.0).to(self.device)

        num_tasks = 1
        rank = 0
        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)

        for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, "Evaluation i2T")):
            topk_sim, topk_idx = sims.topk(k=self.config['k_test'], dim=0)

            encoder_output = image_feats[start + i].repeat(self.config['k_test'], 1, 1).to(self.device)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(self.device)
            output = self.model.text_encoder(text_ids[topk_idx],
                                             attention_mask=text_atts[topk_idx],
                                             encoder_hidden_states=encoder_output,
                                             encoder_attention_mask=encoder_att,
                                             return_dict=True,
                                             )
            score = self.model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_i2t[start + i, topk_idx] = score + topk_sim

        sims_matrix = sims_matrix.t()
        score_matrix_t2i = torch.full((len(texts), image_feats.shape[0]), -100.0).to(self.device)

        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)

        for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, "Evaluation T2i")):

            topk_sim, topk_idx = sims.topk(k=self.config['k_test'], dim=0)
            encoder_output = image_feats[topk_idx].to(self.device)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(self.device)
            output = self.model.text_encoder(text_ids[start + i].repeat(self.config['k_test'], 1),
                                             attention_mask=text_atts[start + i].repeat(self.config['k_test'], 1),
                                             encoder_hidden_states=encoder_output,
                                             encoder_attention_mask=encoder_att,
                                             return_dict=True,
                                             )
            score = self.model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_t2i[start + i, topk_idx] = score + topk_sim

        return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

    def run_scores_batched(self, image_embeds, image_feats, text_embeds, text_ids, text_atts):
        # Should return something with shape (n_tests, n_image_options, n_text_options)
        # Image embeds and all: (n_tests, n_image_options, embed_dim)
        # Text embeds and all: (n_tests, n_text_options, embed_dim)

        # Score matrix should be of the size: (n_tests, n_image_options, n_text_options)

        sims_matrix = torch.einsum('ijk,ilk->ijl', image_embeds, text_embeds)  # (n_tests, n_image_options, n_text_options)

        score_matrix_i2t = torch.full((sims_matrix.shape[0], sims_matrix.shape[1], sims_matrix.shape[2]), -100.0).to(self.device)

        for i, sims in enumerate(sims_matrix):
            for j in range(sims.shape[0]):
                encoder_output = image_feats[i, j].repeat(sims_matrix.shape[2], 1, 1).to(self.device)
                encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(self.device)
                output = self.model.text_encoder(text_ids[i],
                                                 attention_mask=text_atts[i],
                                                 encoder_hidden_states=encoder_output,
                                                 encoder_attention_mask=encoder_att,
                                                 return_dict=True)
                score = self.model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
                score_matrix_i2t[i, j] = score + sims[j]

        sims_matrix = sims_matrix.permute(0, 2, 1)  # (n_tests, n_text_options, n_image_options)
        score_matrix_t2i = torch.full((sims_matrix.shape[0], sims_matrix.shape[1], sims_matrix.shape[2]), -100.0).to(self.device)

        for i, sims in enumerate(sims_matrix):
            for j in range(sims.shape[0]):
                encoder_output = image_feats[i].to(self.device)
                encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(self.device)
                output = self.model.text_encoder(text_ids[i, j].repeat(sims_matrix.shape[2], 1),
                                                 attention_mask=text_atts[i, j].repeat(sims_matrix.shape[2], 1),
                                                 encoder_hidden_states=encoder_output,
                                                 encoder_attention_mask=encoder_att,
                                                 return_dict=True)
                score = self.model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
                score_matrix_t2i[i, j] = score + sims[j]

        return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

    @torch.no_grad()
    def get_retrieval_scores_batched(self, joint_loader):
        """Computes the scores for each image_option / caption_option pair in the joint loader.

        Args:
            joint_loader (DataLoader): batches have "image_options" and "caption_options" fields.
            "image_options" is a list of images, and "caption_options" is a list of captions.

        Returns:
            all_scores: A numpy array containing the scores of the shape NxKxL,
            where N is the number of test cases, K is the number of image options per the test case,
            and L is the number of caption options per the test case.
        """
        t2i_scores, i2t_scores = [], []
        for batch in tqdm(joint_loader):
            image_feats = []
            image_embeds = []
            for i_option in batch["image_options"]:
                image_feat = self.model.visual_encoder(i_option.to(self.device))
                image_embed = self.model.vision_proj(image_feat[:, 0, :])  # B x D
                image_embed = F.normalize(image_embed, dim=-1)

                image_feats.append(image_feat.unsqueeze(1))
                image_embeds.append(image_embed.unsqueeze(1))

            image_feats = torch.cat(image_feats, dim=1)
            image_embeds = torch.cat(image_embeds, dim=1)

            text_ids = []
            text_embeds = []
            text_atts = []

            for c_option in batch["caption_options"]:
                c_option = list(c_option)
                text_input = self.model.tokenizer(c_option, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
                text_output = self.model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
                text_embed = F.normalize(self.model.text_proj(text_output.last_hidden_state[:, 0, :]))

                text_embeds.append(text_embed.unsqueeze(1))
                text_ids.append(text_input.input_ids.unsqueeze(1))
                text_atts.append(text_input.attention_mask.unsqueeze(1))

            text_embeds = torch.cat(text_embeds, dim=1)
            text_ids = torch.cat(text_ids, dim=1)
            text_atts = torch.cat(text_atts, dim=1)
            text_ids[:, :, 0] = self.model.tokenizer.enc_token_id

            s_i2t, s_t2i = self.run_scores_batched(image_embeds, image_feats, text_embeds, text_ids, text_atts)
            t2i_scores.append(s_t2i)
            i2t_scores.append(s_i2t)

        t2i_scores = np.concatenate(t2i_scores, axis=0)  # N x N_t x N_i
        t2i_scores = np.transpose(t2i_scores, (0, 2, 1))  # N x N_i x N_t
        i2t_scores = np.concatenate(i2t_scores, axis=0)  # N x N_i x N_t
        print(t2i_scores.shape, i2t_scores.shape)
        return t2i_scores, i2t_scores


class BLIP2ModelWrapper:
    def _init_(self, root_dir, device, variant="blip2"):
        self.variant = variant
        self.root_dir = root_dir
        # Architectures                  Types
        # ==================================================
        # blip2_opt                      pretrain_opt2.7b, caption_coco_opt2.7b, pretrain_opt6.7b, caption_coco_opt6.7b
        # blip2_t5                       pretrain_flant5xl, caption_coco_flant5xl, pretrain_flant5xxl
        # blip2                          pretrain, coco
        types = {'blip2_t5': 'pretrain_flant5xxl',
                 'blip2': 'pretrained',
                 'blip2_opt': 'pretrain_opt6.7b'}
        model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain",
                                                                           device=device, is_eval=True)

        # model, vis_processors, text_processors = load_model_and_preprocess(name=variant,
        #                                                                    model_type=types[variant],
        #                                                      is_eval=True, device=device)

        # self.config_path = os.path.join(root_dir, f"{self.variant}-config")
        # self.model_path = os.path.join(root_dir, f"{self.variant}.pth")
        # self.bert_config_path = os.path.join(root_dir, "configs", f"{self.variant}_med_config.json")
        # if not (os.path.exists(self.config_path) and os.path.exists(self.model_path) and os.path.exists(
        #         self.bert_config_path)):
        #     self.download()
        #
        # config = yaml.load(open(self.config_path, 'r'), Loader=yaml.Loader)
        # self.config = config
        # self.config['k_test'] = 128
        # config['med_config'] = self.bert_config_path
        # model = blip_retrieval(pretrained=self.model_path, image_size=config['image_size'], vit=config['vit'],
        #                        vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
        #                        queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'],
        #                        med_config=config['med_config'])
        self.vis_processors = vis_processors
        self.text_processors = text_processors
        self.model = model.to(device)
        self.model = self.model.eval().float()
        self.device = device

    def download(self):
        pass
        # print(f"Downloading BLIP model to {self.root_dir}...")
        # model_url = download_urls[self.variant]["model_url"]
        # config_url = download_urls[self.variant]["config_url"]
        # bert_config_url = download_urls[self.variant]["bert_config_url"]
        # os.makedirs(os.path.join(self.root_dir, "configs"), exist_ok=True)
        # subprocess.call(["wget", "-c", model_url, "-O", self.model_path])
        # subprocess.call(["wget", "-c", config_url, "-O", self.config_path])
        # subprocess.call(["wget", "-c", bert_config_url, "-O", self.bert_config_path])

    @torch.no_grad()
    def get_text_embeddings(self, texts, text_batch_size=256):
        # img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        txt = self.text_processors["eval"](texts)
        # num_text = len(texts)
        # text_bs = 256
        # text_ids = []
        # text_embeds = []
        # text_atts = []
        # for i in range(0, num_text, text_bs):
        #     text = texts[i: min(num_text, i + text_bs)]
        #     text_input = self.model.tokenizer(text, padding='max_length', truncation=True, max_length=35,
        #                                       return_tensors="pt").to(self.device)
        #     text_output = self.model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask,
        #                                           mode='text')
        #     text_embed = F.normalize(self.model.text_proj(text_output.last_hidden_state[:, 0, :]))
        #     text_embeds.append(text_embed)
        #     text_ids.append(text_input.input_ids)
        #     text_atts.append(text_input.attention_mask)
        #
        # text_embeds = torch.cat(text_embeds, dim=0)
        # text_ids = torch.cat(text_ids, dim=0)
        # text_atts = torch.cat(text_atts, dim=0)
        # text_ids[:, 0] = self.model.tokenizer.enc_token_id

        return txt

    @torch.no_grad()
    def get_image_embeddings(self, image_loader):
        image_feats = []
        image_embeds = []
        for batch in tqdm(image_loader):
            image = batch["image"]
            image = image.to(self.device)
            image_feat = self.model.visual_encoder(image)
            image_embed = self.model.ln_vision(image_feat[:, 0, :])
            image_embed = F.normalize(image_embed, dim=-1)

            image_feats.append(image_feat.cpu())
            image_embeds.append(image_embed)

        image_feats = torch.cat(image_feats, dim=0)
        image_embeds = torch.cat(image_embeds, dim=0)
        return image_feats, image_embeds

    @torch.no_grad()
    def get_retrieval_scores_dataset(self, loader):
        texts = loader.dataset.text
        metric_logger = MetricLogger(delimiter="  ")

        text_embeds, text_ids, text_atts = self.get_text_embeddings(texts)
        image_feats, image_embeds = self.get_image_embeddings(loader)
        sims_matrix = image_embeds @ text_embeds.t()
        score_matrix_i2t = torch.full((image_embeds.shape[0], len(texts)), -100.0).to(self.device)

        num_tasks = 1
        rank = 0
        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)

        for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, "Evaluation i2T")):
            topk_sim, topk_idx = sims.topk(k=self.config['k_test'], dim=0)

            encoder_output = image_feats[start + i].repeat(self.config['k_test'], 1, 1).to(self.device)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(self.device)
            output = self.model.text_encoder(text_ids[topk_idx],
                                             attention_mask=text_atts[topk_idx],
                                             encoder_hidden_states=encoder_output,
                                             encoder_attention_mask=encoder_att,
                                             return_dict=True,
                                             )
            score = self.model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_i2t[start + i, topk_idx] = score + topk_sim

        sims_matrix = sims_matrix.t()
        score_matrix_t2i = torch.full((len(texts), image_feats.shape[0]), -100.0).to(self.device)

        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)

        for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, "Evaluation T2i")):
            topk_sim, topk_idx = sims.topk(k=self.config['k_test'], dim=0)
            encoder_output = image_feats[topk_idx].to(self.device)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(self.device)
            output = self.model.text_encoder(text_ids[start + i].repeat(self.config['k_test'], 1),
                                             attention_mask=text_atts[start + i].repeat(self.config['k_test'], 1),
                                             encoder_hidden_states=encoder_output,
                                             encoder_attention_mask=encoder_att,
                                             return_dict=True,
                                             )
            score = self.model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_t2i[start + i, topk_idx] = score + topk_sim

        return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

    def run_scores_batched(self, image_embeds, image_feats, text_embeds, text_ids, text_atts):
        # Should return something with shape (n_tests, n_image_options, n_text_options)
        # Image embeds and all: (n_tests, n_image_options, embed_dim)
        # Text embeds and all: (n_tests, n_text_options, embed_dim)

        # Score matrix should be of the size: (n_tests, n_image_options, n_text_options)

        sims_matrix = torch.einsum('ijk,ilk->ijl', image_embeds,
                                   text_embeds)  # (n_tests, n_image_options, n_text_options)

        score_matrix_i2t = torch.full((sims_matrix.shape[0], sims_matrix.shape[1], sims_matrix.shape[2]), -100.0).to(
            self.device)

        for i, sims in enumerate(sims_matrix):
            for j in range(sims.shape[0]):
                encoder_output = image_feats[i, j].repeat(sims_matrix.shape[2], 1, 1).to(self.device)
                encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(self.device)
                output = self.model.text_encoder(text_ids[i],
                                                 attention_mask=text_atts[i],
                                                 encoder_hidden_states=encoder_output,
                                                 encoder_attention_mask=encoder_att,
                                                 return_dict=True)
                score = self.model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
                score_matrix_i2t[i, j] = score + sims[j]

        sims_matrix = sims_matrix.permute(0, 2, 1)  # (n_tests, n_text_options, n_image_options)
        score_matrix_t2i = torch.full((sims_matrix.shape[0], sims_matrix.shape[1], sims_matrix.shape[2]), -100.0).to(
            self.device)

        for i, sims in enumerate(sims_matrix):
            for j in range(sims.shape[0]):
                encoder_output = image_feats[i].to(self.device)
                encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(self.device)
                output = self.model.text_encoder(text_ids[i, j].repeat(sims_matrix.shape[2], 1),
                                                 attention_mask=text_atts[i, j].repeat(sims_matrix.shape[2], 1),
                                                 encoder_hidden_states=encoder_output,
                                                 encoder_attention_mask=encoder_att,
                                                 return_dict=True)
                score = self.model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
                score_matrix_t2i[i, j] = score + sims[j]

        return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

    @torch.no_grad()
    def get_retrieval_scores_batched(self, joint_loader):
        """Computes the scores for each image_option / caption_option pair in the joint loader.

        Args:
            joint_loader (DataLoader): batches have "image_options" and "caption_options" fields.
            "image_options" is a list of images, and "caption_options" is a list of captions.

        Returns:
            all_scores: A numpy array containing the scores of the shape NxKxL,
            where N is the number of test cases, K is the number of image options per the test case,
            and L is the number of caption options per the test case.
        """
        t2i_scores, i2t_scores = [], []
        for batch in tqdm(joint_loader):
            image_feats = []
            image_embeds = []
            for i_option in batch["image_options"]:
                with self.model.maybe_autocast():
                    image_embed = self.model.ln_vision(self.model.visual_encoder(i_option.to(self.device)))
                image_embed = image_embed.float()
                # image_feat = self.model.visual_encoder(i_option.to(self.device))
                # image_embed = self.model.ln_vision(image_feat[:, 0, :])  # B x D
                image_embed = F.normalize(image_embed, dim=-1)
                query_tokens = self.model.query_tokens.expand(image_embed.shape[0], -1, -1)
                # image_embed = image_embed.unsqueeze(1)
                image_atts = torch.ones(image_embed.size()[:-1], dtype=torch.long).to(
                    image_embed.device
                )
                query_output = self.model.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embed,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
                image_feat = F.normalize(
                    self.model.vision_proj(query_output.last_hidden_state), dim=-1
                )

                image_feats.append(image_feat.unsqueeze(1))
                image_embeds.append(image_embed.unsqueeze(1))

            image_feats = torch.cat(image_feats, dim=1)
            image_embeds = torch.cat(image_embeds, dim=1)

            text_ids = []
            text_embeds = []
            text_atts = []

            for c_option in batch["caption_options"]:
                c_option = list(c_option)
                text_input = self.model.tokenizer(c_option, padding='max_length', truncation=True, max_length=35,
                                                  return_tensors="pt").to(self.device)
                text_output = self.model.Qformer.bert(
                    text_input.input_ids,
                    attention_mask=text_input.attention_mask,
                    return_dict=True,
                )
                text_embed = F.normalize(
                    self.model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
                )
                # text_embed = F.normalize(self.model.text_proj(text_output.last_hidden_state[:, 0, :]))

                text_embeds.append(text_embed.unsqueeze(1))
                text_ids.append(text_input.input_ids.unsqueeze(1))
                text_atts.append(text_input.attention_mask.unsqueeze(1))

            text_embeds = torch.cat(text_embeds, dim=1)
            text_ids = torch.cat(text_ids, dim=1)
            text_atts = torch.cat(text_atts, dim=1)
            # text_ids[:, :, 0] = self.model.tokenizer.enc_token_id
            sims_matrix = torch.einsum('ijk,ilk->ijl', image_feats,
                                       text_embeds)
            s_i2t, s_t2i = self.run_scores_batched(image_embeds, image_feats, text_embeds, text_ids, text_atts)
            t2i_scores.append(s_t2i)
            i2t_scores.append(s_i2t)

        t2i_scores = np.concatenate(t2i_scores, axis=0)  # N x N_t x N_i
        t2i_scores = np.transpose(t2i_scores, (0, 2, 1))  # N x N_i x N_t
        i2t_scores = np.concatenate(i2t_scores, axis=0)  # N x N_i x N_t
        print(t2i_scores.shape, i2t_scores.shape)
        return t2i_scores, i2t_scores


class BLIP2HFModelWrapper:
    """
    This class is used instead of training_loops, ....etc.
    Every operation under the model, wrapping of questions, search of nearest results - is done here.
    Methods:
        get_scores_for_captions


    Architectures                  Types
    ==================================================
    blip2_opt                      pretrain_opt2.7b, caption_coco_opt2.7b, pretrain_opt6.7b, caption_coco_opt6.7b
    blip2_t5                       pretrain_flant5xl, caption_coco_flant5xl, pretrain_flant5xxl
    blip2                          pretrain, coco
    """

    def __init__(self, root_dir, device, variant="blip2"):
        """
        __init__ function for BLIP2HFModelWrapper

        Args:
            root_dir (_type_): _description_
            device (_type_): _description_
            variant (str, optional): _description_. Defaults to "blip2".
        """
        self.variant = variant
        self.failed_count = 0
        self.positive_count = 0
        self.negative_count = 0
        self.root_dir = root_dir
        types = {'blip2_t5': 'pretrain_flant5xxl',
                 'blip2': 'pretrained',
                 'blip2_opt': 'pretrain_opt6.7b'}

        # Load the BLIP-2 model
        self.model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b-coco')
        processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b-coco')
        # model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-6.7b')#("Salesforce/blip-image-captioning-base")
        # processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b-coco')

        self.processor = processor
        self.model.to(device)
        # self.model = self.model.eval().float()
        self.device = device

    def download(self):
        pass

    @torch.no_grad()
    def get_scores_for_captions(self, batched_captions: str, processed_imgs: dict, batch_size: int = 1, batched_questions=None):
        """Get the scores for the captions - compute loss for each caption, where the caption is the label

        Args:
            processed_captions (str): _description_
            processed_imgs (dict): _description_
            batch_size (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        scores = torch.zeros(batch_size, 4)
        for b_ind in range(batch_size):
            if batched_questions is not None:
                procecced_question = self.processor(text=batched_questions[b_ind])
                instruction = torch.tensor([procecced_question['input_ids']], device='cuda')
                attention_mask = torch.tensor([procecced_question['attention_mask']], device='cuda')
            for c_ind, t_option in enumerate(batched_captions[b_ind]):
                procecced_caption = self.processor(text=t_option)
                # in order to use the question as the instruction
                if batched_questions is None:
                    instruction = torch.tensor([procecced_caption['input_ids']], device='cuda')
                    attention_mask = torch.tensor([procecced_caption['attention_mask']], device='cuda')
                input_data = {'pixel_values': torch.tensor([processed_imgs['pixel_values'][b_ind]], device='cuda'),
                              'input_ids': instruction,
                              'attention_mask': attention_mask,
                              'labels': torch.tensor([procecced_caption['input_ids']], device='cuda'),
                              }
                out_dict = self.model(**input_data, return_dict=True)
                answer_tokens = self.model.generate(**input_data, max_new_tokens=200)
                answer = self.processor.batch_decode(answer_tokens)
                free_generated_caption = self.model.generate(**{'pixel_values': torch.tensor([processed_imgs['pixel_values'][b_ind]], device='cuda'), "output_scores": True})
                suggested_caption = self.processor.batch_decode(free_generated_caption, skip_special_tokens=True)[0].strip()
                scores[b_ind, c_ind] = out_dict['loss']
        return scores


    @torch.no_grad()
    def answer_by_likelihood_for_captions(self, batched_captions: str, processed_imgs: dict, batch_size: int = 1, batched_questions=None):
        """Get the scores for the captions - compute loss for each caption, where the caption is the label

        Args:
            processed_captions (str): _description_
            processed_imgs (dict): _description_
            batch_size (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        scores = torch.zeros(batch_size, 4)
        for b_ind in range(batch_size):
            if batched_questions is not None:
                procecced_question = self.processor(text=batched_questions[b_ind])
                instruction = torch.tensor([procecced_question['input_ids']], device='cuda')
                attention_mask = torch.tensor([procecced_question['attention_mask']], device='cuda')
            for c_ind, t_option in enumerate(batched_captions[b_ind]):
                procecced_caption = self.processor(text=t_option.strip())
                # in order to use the question as the instruction
                if batched_questions is None:
                    instruction = torch.tensor([procecced_caption['input_ids']], device='cuda')
                    attention_mask = torch.tensor([procecced_caption['attention_mask']], device='cuda')
                input_data = {'pixel_values': torch.tensor([processed_imgs['pixel_values'][b_ind]], device='cuda'),
                              "input_ids":  torch.tensor([procecced_caption['input_ids']], device='cuda'),
                              "attention_mask": torch.tensor([procecced_caption['attention_mask']], device='cuda'),
                              'labels': torch.tensor([procecced_caption['input_ids']], device='cuda'),
                              }
                #   as suggested in the captioning phase of BLIP2 originally., that is a caption of the whole image now
                out_dict = self.model(**input_data, return_dict=True)
                answer_tokens = self.model.generate(**input_data, max_new_tokens=200)
                answer = self.processor.batch_decode(answer_tokens)
                scores[b_ind, c_ind] = out_dict['loss']
        return scores


    @torch.no_grad()
    def answer_question_by_text(self, answers, batched_captions, processed_imgs, batch_size: int = 1):
        """using the text of the free answer find the best matching captions based on the text model

        Args:
            answer (_type_): _description_
            captions (_type_): _description_
        """
        choices = []
        scores = torch.zeros((batch_size, 4), device=self.device)
        for batch_index, (captions, answer, image) in enumerate(zip(batched_captions, answers, processed_imgs["pixel_values"])):
            # looping over the batch size instead of range
            losses = torch.empty(0, device=self.device)
            for caption_ind, text_option in enumerate(captions):
                procecced_caption = self.processor(text=text_option)
                processed_answer = self.processor(text=answer)
                inputs_embeds = self.model.get_input_embeddings()(torch.tensor(procecced_caption['input_ids']).to(self.device))
                inputs_embeds = inputs_embeds.to(self.device).unsqueeze(0)
                # attention_mask = torch.tensor(procecced_caption['attention_mask']).to(self.device).unsqueeze(0)
                # now get only the loss for the generation part
                # this is the Generation of the OPT
                # that is not question answering
                attention_mask = torch.tensor([procecced_caption['attention_mask']], device='cuda')
                input_data = {'pixel_values': torch.tensor([image], device='cuda'),
                              'input_ids': torch.tensor([procecced_caption['input_ids']], device='cuda'),
                              'attention_mask': attention_mask,
                              'labels': torch.tensor([processed_answer['input_ids']], device='cuda'),
                              }
                out_dict = self.model(**input_data, return_dict=True)
                outputs = self.model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_scores=True,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    labels=processed_answer['input_ids']
                )
                losses = torch.cat((losses, out_dict.loss.reshape(-1)))
                scores[batch_index, caption_ind] = out_dict.loss
            choice_for_batch = torch.argmin(losses)
            choices.append(choice_for_batch.item())
        return scores, choices

    @torch.no_grad()
    def get_answer_for_question(self, processed_imgs: dict, question: str, batch_size: int = 1, batched_captions=None):
        """Get Answer to the question asked without the multiple choice options

        Args:
            processed_imgs (dict): the Image embeddings
            question (str): the question asked
            batch_size (int, optional): batched examples. Defaults to 1.

        Returns:
            list: list of answers
        """
        procecced_question = self.processor(text=question)
        answers = []
        for b_ind in range(batch_size):
            for c_ind, t_caption in enumerate(batched_captions[b_ind]):
                preprocceced_option = self.processor(text=t_caption)
                input_data = {'pixel_values': torch.tensor([processed_imgs['pixel_values'][b_ind]], device='cuda'),
                              'input_ids': torch.tensor([procecced_question['input_ids']], device='cuda')[b_ind],
                              'attention_mask': torch.tensor([procecced_question['attention_mask']], device='cuda')[b_ind],
                              'labels': torch.tensor([preprocceced_option['input_ids']], device='cuda')[b_ind],
                              }
                generate_kwargs = {
                    "penalty_alpha": 0.6,
                    "top_k": 4,
                    "output_hidden_states": True
                }
                generate_ids = self.model.generate(**input_data, return_dict=True, **generate_kwargs)
                # decide based on the closest hidden state!
                # for contrastive mode
                #
                # here, one step before the decoder as we are using the free answer
                answer = self.processor.batch_decode(generate_ids, skip_special_tokens=True)
                # there is always another dimension - since we are looping through the batch
                answer = [ans.rstrip().lstrip() for ans in answer]
                answers += answer
            # remove spaces and \n
        return answers

    @torch.no_grad()
    def get_retrieval_scores_dataset(self, loader):
        raise NotImplementedError()

    def open_images(self, data_path):
        try:
            raw_image = Image.open(open(data_path, "rb")).convert("RGB")
        except FileNotFoundError:
            # self.failed_count = self.failed_count + 1
            return None
        except Exception as e:
            print(e)
            return None
        return raw_image

    @torch.no_grad()
    def get_retrieval_scores(self, joint_loader, batch_size=1, verbose: bool = False):
        """Computes the scores for each image_option / caption_option pair in the joint loader.
        That function is kind of the main loop

        Args:
            joint_loader (DataLoader): batches have "image_options" and "caption_options" fields.
            "image_options" is a list of images, and "caption_options" is a list of captions.

        Returns:
            all_scores: A numpy array containing the scores of the shape NxKxL,
            where N is the number of test cases, K is the number of image options per the test case,
            and L is the number of caption options per the test case.
        """
        t2i_scores = np.array([], dtype=np.float64).reshape(0, 4)
        answer2id = {"A": 0, "B": 1, "C": 2, "D": 3}
        results_iterator = []
        with tqdm(
            bar_format="CorrectCount: {postfix} | Elapsed: {elapsed} | {rate_fmt}"
        ) as t:
            for batch in tqdm(joint_loader):
                choices = [batch['choice_a'], batch['choice_b'], batch['choice_c'], batch['choice_d']]
                processed_choices = [[c[question_index] for c in choices] for question_index, question in enumerate(batch['question'])]
                processed_captions = [[f"Question: {question} Answer: {c[question_index]}" for c in choices] for question_index, question in enumerate(batch['question'])]
                question_captions = [f"Question: {question}" for question_index, question in enumerate(batch['question'])]
                # new captions, based on rephrasing the question
                # based on the filtering the new are not null
                choices = [batch['new_1'], batch['new_2'], batch['new_3'], batch['new_4']]
                processed_captions = [[c[question_index] for c in choices] for question_index, question in enumerate(batch['question'])]
                data_paths = [os.path.join(image_dir, x) for x in batch['data_id']]
                raw_images = [self.open_images(x) for x in data_paths]
                try:
                    imgs = self.processor(images=raw_images)
                except Exception as e:
                    self.failed_count = self.failed_count + 1
                    continue

                # since we are working with BS =1 here only iterate over captions
                question_type_ids = [int(x) for x in batch['question_type_id']]
                # results = self.get_scores_for_captions(processed_imgs=imgs, batched_captions=processed_captions, batch_size=batch_size)
                # get answer with only question instruction
                results = self.answer_by_likelihood_for_captions(processed_imgs=imgs, batched_captions=processed_captions, batch_size=batch_size)
                # new method
                # answer by captioning
                answer_by_model = self.get_answer_for_question(processed_imgs=imgs, question=question_captions, batch_size=batch_size, batched_captions=processed_choices)
                # append the artificial answer to the original data
                # results, best_match = self.answer_question_by_text(answers=answer_by_model, batched_captions=processed_choices, processed_imgs=imgs, batch_size=batch_size)
                # end of new method
                results = results.cpu().numpy()
                results_iterator.append(results)
                answer_id = [answer2id[ans] for ans in batch["answer"]]
                real_answer = batch.get(fr"choice_{batch['answer'][0].lower()}")
                if verbose:
                    print(fr"question caption: {question_captions}")
                    print(fr"answer_by_model: {answer_by_model}")
                    print(fr"real_answer {real_answer}")
                indexes = np.argsort(results, axis=1)
                correct = indexes[:, 0] == answer_id
                # the correct is array of shape `(batch_size)`
                self.positive_count += correct.sum()
                self.negative_count += (correct.size - correct.sum())
                acc = (self.positive_count / (len(joint_loader) - self.failed_count))
                wandb.log({"acc (cummulative step)": acc})
                wandb.log({"Negative (step)": self.negative_count})
                wandb.log({"success (step)": self.positive_count})
                wandb.log({"Error (step)": self.failed_count})
                t.postfix = f"Correct: {self.positive_count} Not correct: {self.negative_count} Did not read: {self.failed_count}"
                t.update()

        t2i_scores = np.concatenate(results_iterator, axis=0)  # N x N_t x N_i
        acc = (self.positive_count / (len(joint_loader) - self.failed_count))
        acc_percent = acc * 100.0
        self.acc = acc_percent
        wandb.log({"Error (total)": self.failed_count})
        wandb.log({"accuracy (total)": acc_percent})
        return t2i_scores, acc_percent
