from evaluator_strategies.BLIP2Models import SEEDModelWrapper
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
import torch

class InstructBlipModel(SEEDModelWrapper):
    def __init__(self, root_dir, device, names, variant="InstructBLIP"):
        super().__init__(root_dir, device, names, variant)
        # self.model = InstructBlipPreTrainedModel.from_pretrained('Salesforce/instructblip-flan-t5-xl')
        self.model = InstructBlipForConditionalGeneration.from_pretrained('Salesforce/instructblip-flan-t5-xl').to(device)
        self.processor = InstructBlipProcessor.from_pretrained('Salesforce/instructblip-flan-t5-xl')
        # add setup function / use lightning :-)
        

class InstructBlipBaseline(InstructBlipModel):
    def answer(self, batched_captions: str, processed_imgs: dict, batched_questions: str, batch_size: int = 1):
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
            processed_question = f"""Question: {batched_questions[b_ind]}\nAnswer:"""
            procecced_question = self.processor(text=processed_question,return_tensors="pt", padding="longest").to(self.device)
            # use the builtin query tokens
            input_tokenized = procecced_question
            query_tokens = self.model.query_tokens.expand(batch_size, -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)
            Qformer_atts = torch.cat([query_atts, procecced_question.qformer_attention_mask.to(self.device)], dim=1)
            
            pixel_values = torch.tensor(processed_imgs.pixel_values[b_ind]).to(self.device)
            
            image_embeds = self.model.vision_model(pixel_values.unsqueeze(0))[0]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
            
            query_output = self.model.qformer(
                input_ids=input_tokenized.qformer_input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_t5 = self.model.language_projection(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(self.device)


            encoder_atts = torch.cat([atts_t5, input_tokenized.attention_mask], dim=1)
        
            # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            inputs_embeds = self.model.get_input_embeddings()(input_tokenized.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            for c_ind, choice in enumerate(batched_captions[b_ind]):
                output_tokenized = self.processor(text=choice, return_tensors="pt", padding="longest", truncation=True).to(self.device)
                targets = output_tokenized.input_ids.masked_fill(
                output_tokenized.input_ids == self.processor.tokenizer.pad_token_id, -100
                )
                out_dict = self.model.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=encoder_atts,
                    labels=targets,
                    return_dict=True)
                scores[b_ind, c_ind] = out_dict['loss']
        return scores



class InstructBlipAnswerByRephrasing(InstructBlipModel):
    """Using the rephrasing strategy for instructBlip model inputs.
    The reason there is a different rephrasing class for each model is the ability to define where
    each input (new_1,...new_4) will be inserted in the model inputes

    Args:
        SEEDModelWrapper (_type_): _description_
    """
    
    def answer(self, batched_captions: str, processed_imgs: dict, batched_questions: str, batch_size: int = 1):
        """Answer the question based on the rephrased captions gave in batched_catpions

        Args:
            batched_captions (str): _description_
            processed_imgs (dict): _description_
            batched_questions (str): _description_
            batch_size (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """        
        scores = torch.zeros(batch_size, 4)
        for b_ind in range(batch_size):
            insruction = "An image that shows"
            procecced_instruction = self.processor(text=insruction, return_tensors="pt", padding="longest").to(self.device)
            # use the builtin query tokens
            input_tokenized = procecced_instruction
            query_tokens = self.model.query_tokens.expand(batch_size, -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)
            Qformer_atts = torch.cat([query_atts, procecced_instruction.qformer_attention_mask.to(self.device)], dim=1)
            
            pixel_values = torch.tensor(processed_imgs.pixel_values[b_ind]).to(self.device)
            
            image_embeds = self.model.vision_model(pixel_values.unsqueeze(0))[0]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
            
            query_output = self.model.qformer(
                input_ids=input_tokenized.qformer_input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_t5 = self.model.language_projection(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(self.device)


            encoder_atts = torch.cat([atts_t5, input_tokenized.attention_mask], dim=1)
        
            # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            inputs_embeds = self.model.get_input_embeddings()(input_tokenized.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            for c_ind, choice in enumerate(batched_captions[b_ind]):
                output_tokenized = self.processor(text=choice, return_tensors="pt", padding="longest", truncation=True).to(self.device)
                targets = output_tokenized.input_ids.masked_fill(
                output_tokenized.input_ids == self.processor.tokenizer.pad_token_id, -100
                )
                out_dict = self.model.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=encoder_atts,
                    labels=targets,
                    return_dict=True)
                scores[b_ind, c_ind] = out_dict['loss']
        return scores


class IntructBlipEvalByHub(InstructBlipModel):
    """
    IntructBlipEvalByHub This class uses the BlipForConditionalGeneration model to calculate the loss as a black box.
    """
    def answer(self, batched_captions: str, processed_imgs: dict, batch_size: int = 1, batched_questions=None, *args, **kwargs):
        scores = torch.zeros(batch_size, 4)
        for b_ind in range(batch_size):
            insruction = "An image that shows"
            processed_instruction = self.processor(text=insruction, return_tensors="pt", padding="longest").to(self.device)
            # use the builtin query tokens
            query_tokens = self.model.query_tokens.expand(batch_size, -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.device)
            
            pixel_values = torch.tensor(processed_imgs.pixel_values[b_ind]).to(self.device).unsqueeze(0)
            
            # image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
            # with torch.cuda.amp.autocast(dtype=torch.bfloat16):

            for c_ind, choice in enumerate(batched_captions[b_ind]):
                output_tokenized = self.processor(text=choice, return_tensors="pt", padding="longest", truncation=True).to(self.device)
                out_dict = self.model.forward(
                    pixel_values=pixel_values,
                    qformer_input_ids=processed_instruction.qformer_input_ids,
                    qformer_attention_mask=processed_instruction.qformer_attention_mask,
                    input_ids=output_tokenized.input_ids,
                    attention_mask=output_tokenized.attention_mask,
                    return_dict=True,
                    output_attentions=True,
                    output_hidden_states=True,
                    # decoder_attention_mask=encoder_atts,
                    labels=output_tokenized.input_ids
                )
                # verify if the learned tokens in that case are being concatinated to the input
                # targets = output_tokenized.input_ids.masked_fill(
                # output_tokenized.input_ids == self.processor.tokenizer.pad_token_id, -100
                # )
                scores[b_ind, c_ind] = out_dict['loss']
        return scores