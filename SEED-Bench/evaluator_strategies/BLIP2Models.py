from re import I
from PIL import Image
import wandb
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# from .blip_utils.blip_retrieval import blip_retrieval
# from .blip_utils.utils import MetricLogger
from lavis.models import load_model_and_preprocess
from transformers import Blip2ForConditionalGeneration, Blip2Processor, InstructBlipPreTrainedModel,InstructBlipForConditionalGeneration,InstructBlipProcessor
import pandas as pd
import json
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


class SEEDModelWrapper:
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

    def __init__(self, root_dir, device, names=None, task_name: str = None ,variant="blip2"):
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
        self.task_name = task_name

        # Load the BLIP-2 model
        # remove this for later use!
        self.model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b-coco')
        processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b-coco')
        # self.model = Blip2Model.from_pretrained('Salesforce/blip2-opt-2.7b-coco')
        # model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-6.7b')#("Salesforce/blip-image-captioning-base")
        # processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b-coco')
        self.names = names if names is not None else ["choice_a", "choice_b", "choice_c", "choice_d"]
        self.processor = processor
        self.model.to(device)
        self.device = device

    def download(self):
        pass

    def batch4choices(self, batch):
        if self.names:
            # for name in self.names:
            #     yield batch.get(name)
            return [batch.get(name) for name in self.names]
        else:
            raise NotImplementedError("Or define self.names / another method for getting the choices")

    @torch.no_grad()
    def answer(self, batched_captions: str, processed_imgs: dict, batch_size: int = 1, batched_questions=None, *args, **kwargs):
        """Get the scores for the captions - compute loss for each caption, where the caption is the label

        NOTE: gradient is not computed here, as well as in the subclasses

        Args:
            processed_captions (str): _description_
            processed_imgs (dict): _description_
            batch_size (int, optional): _description_. Defaults to 1.

        Raises: NotImplementedError: use child classes
        """
        raise NotImplementedError("That is an interface method, please use the child class for answering logic")

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
    def get_retrieval_scores(self, joint_loader, batch_size=1, total_examples_for_task: int = -1, verbose: bool = False):
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
        global_answers_list = []
        index = []  #ugly solution, use distributed datasets map later on 
        results_iterator = {}
        with tqdm(
            bar_format="CorrectCount: {postfix} | Elapsed: {elapsed} | {rate_fmt}"
        ) as t:
            for batch in tqdm(joint_loader):
                choices = self.batch4choices(batch)
                # processed_choices = [[c[question_index] for c in choices] for question_index, question in enumerate(batch['question'])]
                processed_captions = [[c[question_index] for c in choices] for question_index, question in enumerate(batch['question'])]
                processed_questions = batch['question']  # the processing is now done within the self.answer method
                # new captions, based on rephrasing the question
                # based on the filtering the new are not null
                data_paths = [os.path.join(image_dir, x) for x in batch['data_id']]
                raw_images = [self.open_images(x) for x in data_paths]
                try:
                    imgs = self.processor(images=raw_images)
                except Exception as e:
                    self.failed_count = self.failed_count + 1
                    continue
                # results = self.get_scores_for_captions(processed_imgs=imgs, batched_captions=processed_captions, batch_size=batch_size)
                # get answer with only question instruction
                results = self.answer(processed_imgs=imgs, batched_captions=processed_captions, batched_questions=processed_questions, batch_size=batch_size)
                # new method
                # answer by captioning
                # global_answers_list += answers_list
                # end of new method
                results = results.cpu().numpy()
                question_id = batch["question_id"][0]
                index.append(question_id)
                results_iterator[batch["question_id"][0]] = results
                # Batching!
                # update the results base on the ID of the answer
                # save the results for the question
                batch["prediction"] = results
                answer_id = [answer2id[ans] for ans in batch["answer"]]
                real_answer = batch.get(fr"choice_{batch['answer'][0].lower()}")
                indexes = np.argsort(results, axis=-1)
                correct = indexes[:, 0] == answer_id
                # the correct is array of shape `(batch_size)`
                self.positive_count += correct.sum()
                self.negative_count += (correct.size - correct.sum())
                acc = (self.positive_count / total_examples_for_task)
                wandb.log({"acc (step)": acc})
                wandb.log({"Negative (step)": self.negative_count})
                wandb.log({"success (step)": self.positive_count})
                wandb.log({"Error (step)": self.failed_count})
                t.postfix = f"Correct: {self.positive_count} Not correct: {self.negative_count} Did not read: {self.failed_count}"
                t.update()
        acc = (self.positive_count / total_examples_for_task)
        acc_percent = acc * 100.0
        self.acc = acc_percent
        t2i_scores = np.concatenate(list(results_iterator.values()), axis=0)
        pd.DataFrame(t2i_scores).to_csv(rf"scores_for_{self.task_name}_under_{self.__class__.__name__}.csv", index=index)
        pd.DataFrame(global_answers_list).to_csv("answers.csv")
        # how many of these answers are the same?
        # TODO change the zero shot captioning loss for the text generation loss on a single label.
        # wandb.log({"Error (total)": self.failed_count})
        # the metric is already reported
        wandb.log({"accuracy (total)": acc_percent})
        return t2i_scores, acc_percent


class Blip2AnswerByConcatination(SEEDModelWrapper):
    """
    Blip2AnswerByConcatination _summary_

    Args:
        BLIP2HFModelWrapper (_type_): _description_
    """

    def __init__(self, root_dir, device, names, variant="blip2"):
        super().__init__(root_dir, device, names, variant)

    def answer(self, batched_captions: str, processed_imgs: dict, batch_size: int = 1, batched_questions=None):
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


class Blip2AnswerByFreeText(SEEDModelWrapper):

    @torch.no_grad()
    def answer(self, answers, batched_captions, processed_imgs, batch_size: int = 1):
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


class Blip2AnswerByQuestionRephrasing(SEEDModelWrapper):
    """
    Blip2AnswerByRephrasing _summary_

    Args:
        BLIP2HFModelWrapper (_type_): _description_
    """
    def batch4choices(self, batch):
        return [batch.get(name) for name in self.names]

    @torch.no_grad()
    def answer(self, batched_captions: str, processed_imgs: dict, batch_size: int = 1, batched_questions=None):
        """Get the scores for the captions - compute loss for each caption, where the caption is the label

        Args:
            processed_captions (str): _description_
            processed_imgs (dict): _description_
            batch_size (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        # answers for statistics / histogram
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
                              "input_ids": torch.tensor([procecced_caption['input_ids']], device='cuda'),
                              "attention_mask": torch.tensor([procecced_caption['attention_mask']], device='cuda'),
                              'labels': torch.tensor([procecced_caption['input_ids']], device='cuda'),
                              }
                input_data_b = {'pixel_values': torch.tensor([processed_imgs['pixel_values'][b_ind]], device='cuda'),
                                'labels': torch.tensor([procecced_caption['input_ids']], device='cuda')
                                }
                #   as suggested in the captioning phase of BLIP2 originally., that is a caption of the whole image now
                # return scores for that
                out_dict = self.model(**input_data, return_dict=True)
                answer_tokens = self.model.generate(
                    **input_data_b,
                    return_dict=True,
                    output_hidden_states=True,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                # the generate scores are based on the logits of the next token, as long as the logits_processor is none in that case.
                # consider using the self.lm_head learned projection in that point as in the model forward
                # Since the generate -> model_generate -> greedy search -> text_model_forward, the logits are based on the lm_head and they are the same as model.forward
                # these logits are called now "scores", and since no logits_processor exists, they do not lean on history - they are the exact model logits as we know them.
                # now we may use them to compute the loss as we used to do in the model_forward
                logits = torch.cat(answer_tokens.scores, dim=1)
                answer = self.processor.batch_decode(answer_tokens.sequences)
                scores[b_ind, c_ind] = out_dict['loss']
        return scores



class Blip2AnswerByClassic(SEEDModelWrapper):
    """
    Blip2AnswerByClassic That is SEED original evaluation strategy

    Args:
        BLIP2HFModelWrapper (_type_): _description_
    """
    def answer(self, batched_captions: str, processed_imgs: dict, batched_questions: list,batch_size: int = 1):
        # answers for statistics / histogram
        scores = torch.zeros(batch_size, 4)
        for b_ind in range(batch_size):
            procecced_question = self.processor(text=batched_questions[b_ind])
            instruction = torch.tensor([procecced_question['input_ids']], device='cuda')
            attention_mask = torch.tensor([procecced_question['attention_mask']], device='cuda')
            for c_ind, t_option in enumerate(batched_captions[b_ind]):
                procecced_caption = self.processor(text=t_option.strip())
                # in order to use the question as the instruction
                input_data = {'pixel_values': torch.tensor([processed_imgs['pixel_values'][b_ind]], device='cuda'),
                              "input_ids": instruction,
                              "attention_mask": attention_mask,
                              'labels': torch.tensor([procecced_caption['input_ids']], device='cuda'),
                              }
                input_data_b = {'pixel_values': torch.tensor([processed_imgs['pixel_values'][b_ind]], device='cuda'),
                                'labels': torch.tensor([procecced_caption['input_ids']], device='cuda')
                                }
                #   as suggested in the captioning phase of BLIP2 originally., that is a caption of the whole image now
                # return scores for that
                out_dict = self.model(**input_data, return_dict=True)
                answer_tokens = self.model.generate(
                    **input_data_b,
                    return_dict=True,
                    output_hidden_states=True,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                # the generate scores are based on the logits of the next token, as long as the logits_processor is none in that case.
                # consider using the self.lm_head learned projection in that point as in the model forward
                # Since the generate -> model_generate -> greedy search -> text_model_forward, the logits are based on the lm_head and they are the same as model.forward
                # these logits are called now "scores", and since no logits_processor exists, they do not lean on history - they are the exact model logits as we know them.
                # now we may use them to compute the loss as we used to do in the model_forward
                scores[b_ind, c_ind] = out_dict['loss']
        return scores
