# Copyright (c) SLAM-LLM contributors.
# Licensed under the MIT License.
#
# Modifications by Sergio Burdisso (Idiap Research Institute © 2025):
#   - Enhanced prompt parsing to support <speech>, <p:N>, and <p:TEXT> tokens
#   - Added flexible speech token placement anywhere in the prompt
#   - Added support for learnable token insertion and text-initialized learnable tokens
# These modifications are licensed under the MIT License.

import re
import json
import copy
import torch
import logging
import whisper
import numpy as np


logger = logging.getLogger(__name__)

TOKEN_SPEECH = "<speech>"


class SpeechDatasetJsonl(torch.utils.data.Dataset):

    def __init__(
        self, dataset_config, tokenizer=None, processor=None, split="train", llm_name="vicuna-7b-v1.5"
    ):
        global stoken_ix
        super().__init__()
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.processor = processor
        self.llm_name = llm_name
        # data_parallel_size = dist.get_world_size()
        data_parallel_size = 1

        # self.data_list = contents
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.AUDIO_TOKEN_ID = -1
        self.mel_size = dataset_config.get(
            "mel_size", 80
        )  # 80 for whisper large v1 and v2, 128 for large v3
        # self.prompt_library = [
        #     "Begin by converting the spoken words into written text. ",
        #     "Can you transcribe the speech into a written format? ",
        #     "Focus on translating the audible content into text. ",
        #     "Transcribe the speech by carefully listening to it. ",
        #     "Would you kindly write down the content of the speech? ",
        #     "Analyze the speech and create a written transcription. ",
        #     "Engage with the speech to produce a text-based version. ",
        #     "Can you document the speech in written form? ",
        #     "Transform the spoken words into text accurately. ",
        #     "How about putting the speech's content into writing? "
        # ]
        self.prompt = dataset_config.get("prompt", None)
        if self.prompt is None:
            # self.prompt = random.choice(self.prompt_library)
            # self.prompt = "Transcribe speech to text. "
            self.prompt = "Transcribe speech to text. Output the transcription directly without redundant content. Ensure that the output is not duplicated. "
        # if "llama3" in llm_name:
        #     self.prompt_template = [
        #                             {"role": "system",
        #                             "content": "You are a helpful assistant, expertized in transcribing speech to text.",
        #                             },
        #                             {
        #                             "role": "user",
        #                             "content":"Transcribe speech to text: \n",
        #                             }
        #                         ]
        #     #self.prompt_template = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)  
        #     #self.prompt_template = "USER: {}\n ASSISTANT:"
        #     self.prompt = self.processor.apply_chat_template(self.prompt_template, add_generation_prompt=True, tokenize=False)
        # else:
        self.prompt_template = dataset_config.get("prompt_template", None)
        if self.prompt_template:
            self.prompt = self.prompt_template.format(prompt=self.prompt)

        # If p-prompt embeddings initialization...
        logger.info(f"Input prompt: {self.prompt}")
        p_token = dataset_config.prompt_token
        re_token = f"{p_token[:-1]}:.+?{p_token[-1]}"
        m = re.search(re_token, self.prompt, flags=re.IGNORECASE|re.DOTALL)
        if m:
            logger.info(f"  Embedding initialization detected.")
            prompt_new = []
            for piece in re.split(f"({re_token})", self.prompt, flags=re.IGNORECASE|re.DOTALL):
                if re.match(re_token, piece, flags=re.IGNORECASE|re.DOTALL):
                    init_text = re.match(f"{p_token[:-1]}:(.+?){p_token[-1]}", piece, flags=re.IGNORECASE|re.DOTALL).group(1)
                    input_ids = tokenizer(init_text, add_special_tokens=False)["input_ids"]
                    logger.info(f"    -> Segment '{init_text}' initialized with embeddings for {input_ids} tokens.")
                    prompt_new.append(p_token * len(input_ids))
                else:
                    prompt_new.append(piece)
            self.prompt = "".join(prompt_new)
            logger.info(f"New input prompt: {self.prompt}")

        if dataset_config.prompt_embeddings_path:
            stoken_ix = 0
            def token_number(_):
                global stoken_ix
                stoken_ix = stoken_ix + 1
                return f"<p{stoken_ix - 1}>"
            self.prompt = re.sub(p_token, token_number, self.prompt)
            logger.info(f"Prompt embeddings path detected -> New input prompt: {self.prompt}")
            new_tokens = [f"<p{ix}>" for ix in range(stoken_ix)]
            for new_token in new_tokens:
                # one by one to force the right order, to match the indexes in the Embedding matrix
                tokenizer.add_special_tokens({'additional_special_tokens': [new_token]})
            logger.info(f"Adding new tokens to tokenizer: {new_tokens}")

        self.speech_token_ix = 0
        # If prompt_template contains <speech>, set the speech tokens flag and add token to tokenizer
        if TOKEN_SPEECH not in self.prompt:
            logger.info(f"No speech token ('{TOKEN_SPEECH}') was found in the input prompt. "
                        "Speech embeddings will be automatically prepend to the input.")
            self.prompt_ids = self.tokenizer.encode(self.prompt)
            self.speech_token_prefix = True
        else:
            tokenizer.add_special_tokens({'additional_special_tokens': [TOKEN_SPEECH]})
            speech_token_id = tokenizer.get_vocab()[TOKEN_SPEECH]
            self.prompt_ids = self.tokenizer.encode(self.prompt)
            self.speech_token_ix = self.prompt_ids.index(speech_token_id)
            del self.prompt_ids[self.speech_token_ix]
            self.speech_token_prefix = False
            logger.info(f"Speech token found ('{TOKEN_SPEECH}') at index {self.speech_token_ix}.")

        self.answer_template = "{}"
        self.fix_length_audio = dataset_config.get("fix_length_audio", -1)
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.normalize = dataset_config.get("normalize", False)
        self.target_lowercase = dataset_config.get("target_lowercase", False)
        self.input_type = dataset_config.get("input_type", None)
        assert self.input_type in ["raw", "mel"], "input_type must be one of [raw, mel]"

        self.data_list = []
        if split == "train":
            with open(dataset_config.train_data_path, encoding="utf-8") as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    self.data_list.append(data_dict)
        else:
            with open(dataset_config.val_data_path, encoding="utf-8") as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    self.data_list.append(data_dict)

        # # debug
        # with open(dataset_config.train_data_path, encoding='utf-8') as fin:
        #         for line in fin:
        #             data_dict = json.loads(line.strip())
        #             self.data_list.append(data_dict)
        # if split == "train":
        #     self.data_list = self.data_list[:80]
        # else:
        #     self.data_list = self.data_list[80:100]

    def get_source_len(self, data_dict):
        return data_dict["source_len"]

    def get_target_len(self, data_dict):

        return data_dict["target_len"] if "target_len" in data_dict else 0

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_dict = self.data_list[index]
        audio_path = data_dict.get("source")
        target = data_dict.get("target", None)
        task = data_dict.get("prompt", "ASR")
        key = data_dict.get("key", None)

        audio_raw = whisper.load_audio(audio_path,sr=16000) ## sr added to make sure all is in 16KHz
        if self.input_type == "raw":
            audio_raw = torch.from_numpy(audio_raw)
            if self.normalize:
                audio_raw = torch.nn.functional.layer_norm(audio_raw, audio_raw.shape)
            audio_length = len(audio_raw) // 320  # ad-hoc for fairseq 320x downsample
            audio_length = audio_length // 5  # ad-hoc for 5x fc downsample
        elif self.input_type == "mel":
            audio_raw = whisper.pad_or_trim(audio_raw)
            # audio_raw = np.concatenate((np.zeros(random.randint(0, 16000)), audio_raw, np.zeros(random.randint(0, 16000)))).astype(audio_raw.dtype)[:16000*30]
            audio_mel = whisper.log_mel_spectrogram(
                audio_raw, n_mels=self.mel_size
            ).permute(1, 0)
            audio_length = (
                audio_mel.shape[0] + 1
            ) // 2  # ad-hoc for whisper for 2x downsample from mel to feats
            audio_length = audio_length // 5  # ad-hoc for 5x fc downsample
            # audio_length = calculate_output_length_1d(audio_length, 5, 5, 0) # ad-hoc for 5x cov1d downsample
        if self.fix_length_audio > 0:
            audio_length = self.fix_length_audio

        prompt_length = len(self.prompt_ids)

        if self.inference_mode:
            example_ids = self.prompt_ids[:self.speech_token_ix] + ([self.AUDIO_TOKEN_ID] * audio_length) + self.prompt_ids[self.speech_token_ix:]
            example_ids = torch.tensor(example_ids, dtype=torch.int64)
            example_mask = example_ids.ge(-1)  # [True,True]

            return {
                "input_ids": example_ids,
                "attention_mask": example_mask,
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_length": audio_length,
                "key": key,
                "target": target,
                "prompt_length": prompt_length,
            }

        example_ids = self.tokenizer.encode(self.prompt + self.answer_template.format(target))  # [prompt,answer]
        if not self.speech_token_prefix:
            del example_ids[self.speech_token_ix]
        example_ids = example_ids[:self.speech_token_ix] + ([self.AUDIO_TOKEN_ID] * audio_length) + example_ids[self.speech_token_ix:] + [self.tokenizer.eos_token_id]
        example_ids = torch.tensor(example_ids, dtype=torch.int64)

        labels_ids = copy.deepcopy(example_ids)  # [audio,prompt,answer,eos]
        labels_ids[: audio_length + prompt_length] = -1  # [-1,-1,answer,eos];
        example_mask = example_ids.ge(-1)  # FIX(GZF): [True,True,True,True]

        label_mask = labels_ids.ge(0)  # [False,False,True,True]
        example_ids[~example_mask] = 0  # [audio,prompt,answer,eos]
        labels_ids[~label_mask] = self.IGNORE_INDEX  # [-100,-100,answer,eos]

        return {
            "input_ids": example_ids,
            "labels": labels_ids,
            "attention_mask": example_mask,
            "audio": audio_raw if self.input_type == "raw" else None,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_length": audio_length,
            "prompt_length": prompt_length,
            "target": target,
        }

    def pad(self, sequence, max_length, padding_idx=0):
        if isinstance(sequence, (int, list, tuple)):
            if len(sequence) < max_length:
                sequence = sequence + [padding_idx] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, torch.Tensor):
            if len(sequence) < max_length:
                sequence = torch.cat(
                    (
                        sequence,
                        torch.full(
                            ([max_length - len(sequence)] + list(sequence.size())[1:]),
                            padding_idx,
                        ),
                    )
                )
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, np.ndarray):
            if len(sequence) < max_length:
                sequence = np.concatenate(
                    (
                        sequence,
                        np.full(
                            (max_length - len(sequence),) + sequence.shape[1:],
                            padding_idx,
                        ),
                    )
                )
            else:
                sequence = sequence[:max_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence

    @classmethod
    def padding(cls, sequence, padding_length, padding_idx=0, padding_side="right"):
        if isinstance(sequence, (int, list, tuple)):
            if padding_length >= 0:
                sequence = sequence + [padding_idx] * padding_length
            else:
                sequence = sequence[:padding_length]
        elif isinstance(sequence, torch.Tensor):
            if sequence.ndimension() == 2:
                if padding_length >= 0:
                    sequence = torch.nn.functional.pad(sequence, (0, padding_length))
                else:
                    sequence = sequence[:, :padding_length]
            else:
                if padding_length >= 0:
                    if padding_side == "left":
                        sequence = torch.cat(
                            (
                                torch.full(
                                    ([padding_length] + list(sequence.size())[1:]),
                                    padding_idx,
                                ),
                                sequence,
                            )
                        )
                    else:
                        sequence = torch.cat(
                            (
                                sequence,
                                torch.full(
                                    ([padding_length] + list(sequence.size())[1:]),
                                    padding_idx,
                                ),
                            )
                        )
                else:
                    sequence = sequence[:padding_length]
        elif isinstance(sequence, np.ndarray):
            if padding_length >= 0:
                sequence = np.concatenate(
                    (
                        sequence,
                        np.full((padding_length,) + sequence.shape[1:], padding_idx),
                    )
                )
            else:
                sequence = sequence[:padding_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence

    def collator(self, samples):
        assert samples is not None
        input_prompt_lengths = [
            s["audio_length"] + s["prompt_length"] for s in samples
        ]  # [120, 48, 82, 42]
        input_answer_lengths = [
            len(s["input_ids"]) - s["audio_length"] - s["prompt_length"]
            for s in samples
        ]  # [0, 0, 0, 0]

        input_prompt_max_length = max(input_prompt_lengths)
        input_answer_max_length = max(input_answer_lengths)

        input_ids = torch.stack(
            [
                self.padding(
                    self.padding(
                        samples[index]["input_ids"],
                        input_prompt_max_length - input_prompt_lengths[index],
                        self.tokenizer.pad_token_id,
                        padding_side="left",
                    ),
                    input_answer_max_length - input_answer_lengths[index],
                    self.tokenizer.pad_token_id,
                )
                for index in range(len(samples))
            ]
        )

        attention_mask = torch.stack(
            [
                self.padding(
                    self.padding(
                        samples[index]["attention_mask"],
                        input_prompt_max_length - input_prompt_lengths[index],
                        False,
                        padding_side="left",
                    ),
                    input_answer_max_length - input_answer_lengths[index],
                    False,
                )
                for index in range(len(samples))
            ]
        )

        if self.input_type == "raw":
            audio_raw_max_length = max([s["audio"].shape[0] for s in samples])
            audio_raw = torch.stack(
                [self.pad(s["audio"], audio_raw_max_length, 0) for s in samples]
            )
            audio_mask = torch.zeros(len(samples), audio_raw_max_length)
            for line, sample in enumerate(samples):
                audio_mask[line, : sample["audio"].shape[0]] = 1
        elif self.input_type == "mel":
            audio_mel_max_length = max([s["audio_mel"].shape[0] for s in samples])
            audio_mel = torch.stack(
                [self.pad(s["audio_mel"], audio_mel_max_length, 0) for s in samples]
            )
            audio_mel_post_mask = torch.zeros(
                len(samples), (audio_mel_max_length + 1) // 2
            )  # ad-hoc for whisper for 2x downsample from mel to feats
            for line, sample in enumerate(samples):
                audio_mel_post_mask[line, : (sample["audio_mel"].shape[0] + 1) // 2] = 1

        # later, in the forward pass, embeddings masked by this mask will be replaced by speech embeddings
        modality_mask = input_ids == self.AUDIO_TOKEN_ID

        position_ids = None
        if attention_mask is not None:
            # To use the right position embedding when left padding is used
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        if self.inference_mode:
            keys = [s["key"] for s in samples]
            targets = [s["target"] for s in samples]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mask": audio_mask if self.input_type == "raw" else None,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_mel_post_mask": (
                    audio_mel_post_mask if self.input_type == "mel" else None
                ),
                "modality_mask": modality_mask,
                "keys": keys,
                "targets": targets,
            }

        labels = torch.stack(
            [
                self.padding(
                    self.padding(
                        samples[index]["labels"],
                        input_prompt_max_length - input_prompt_lengths[index],
                        self.IGNORE_INDEX,
                        padding_side="left",
                    ),
                    input_answer_max_length - input_answer_lengths[index],
                    self.IGNORE_INDEX,
                )
                for index in range(len(samples))
            ]
        )
        targets = [s["target"] for s in samples]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "audio": audio_raw if self.input_type == "raw" else None,
            "audio_mask": audio_mask if self.input_type == "raw" else None,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_mel_post_mask": (
                audio_mel_post_mask if self.input_type == "mel" else None
            ),
            "modality_mask": modality_mask,
            "targets": targets,
        }


def get_speech_dataset(dataset_config, tokenizer, processor, split, llm_name="vicuna-7b-v1.5"):
    dataset = SpeechDatasetJsonl(dataset_config, tokenizer, processor, split, llm_name)

    return dataset
