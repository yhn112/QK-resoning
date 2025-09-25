from typing import List, Tuple, Optional
import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


class LocalModel:
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-Math-7B-Instruct",
        device_map: str = "auto",
        dtype: Optional[torch.dtype] = torch.bfloat16,   
        trust_remote_code: bool = True,
        enable_flash_attn: bool = True,
    ):
        hf_token = os.environ.get("HF_TOKEN", None)

        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            token=hf_token,
        )

        load_kwargs = dict(
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        if enable_flash_attn:
            load_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            token=hf_token,
            **load_kwargs
        )

        # Ensure a pad token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # EOS set (cover common chat end tokens)
        self.eos_token_id = self.tokenizer.eos_token_id
        if isinstance(self.eos_token_id, int):
            self.eos_token_id = [self.eos_token_id]
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            for tok in ["<|im_end|>", "<|endoftext|>"]:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid != self.tokenizer.unk_token_id and tid not in self.eos_token_id:
                    self.eos_token_id.append(tid)

        self.model.config.use_cache = True

    def _format_prompt(self, user_text: str) -> str:
        messages = [{"role": "user", "content": user_text}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


    def _decode_generated(self, out_ids: torch.Tensor, input_len: int) -> Tuple[str, int]:
        pad_id = self.tokenizer.pad_token_id
        eos_ids = set(self.eos_token_id)
        gen_ids = out_ids[input_len:]
        gen_len = gen_ids.shape[0]
        for j, tok in enumerate(gen_ids.tolist()):
            if tok in eos_ids:
                gen_len = j + 1
                break
            if pad_id is not None and tok == pad_id:
                gen_len = j
                break
        true_gen_ids = gen_ids[:gen_len]
        text = self.tokenizer.decode(true_gen_ids, skip_special_tokens=True)
        return text, int(true_gen_ids.shape[0])

    def _generate_batch(
        self,
        inputs,
        input_len: int,
        num_seqs: int,
        gen_cfg: GenerationConfig,
    ) -> Tuple[List[str], List[int]]:
        """
        Try to generate `num_seqs` in one batched call (num_return_sequences=num_seqs).
        If CUDA OOM, fall back to sequential generation num_seqs × 1.
        """
        texts, counts = [], []

        try:
            cfg = GenerationConfig.from_dict(gen_cfg.to_dict())
            cfg.num_return_sequences = num_seqs

            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    generation_config=cfg,
                    return_dict_in_generate=False,
                )

            out_cpu = out.to("cpu")
            del out
            torch.cuda.empty_cache()

            for i in range(out_cpu.shape[0]):
                text, cnt = self._decode_generated(out_cpu[i], input_len)
                texts.append(text)
                counts.append(cnt)

            del out_cpu
            gc.collect()
            torch.cuda.empty_cache()
            return texts, counts

        except torch.cuda.OutOfMemoryError:

            torch.cuda.empty_cache()
            seq_cfg = GenerationConfig.from_dict(gen_cfg.to_dict())
            seq_cfg.num_return_sequences = 1

            for _ in range(num_seqs):
                with torch.inference_mode():
                    out = self.model.generate(
                        **inputs,
                        generation_config=seq_cfg,
                        return_dict_in_generate=False,
                    )
                out_cpu = out[0].to("cpu")
                del out
                torch.cuda.empty_cache()

                text, cnt = self._decode_generated(out_cpu, input_len)
                texts.append(text)
                counts.append(cnt)

                del out_cpu
                if hasattr(self.model, "past_key_values"):
                    self.model.past_key_values = None
                gc.collect()
                torch.cuda.empty_cache()

            return texts, counts

    def generate(
        self,
        prompt_text: str,
        n: int = 1,                           
        temperature: float = 0.0,
        repetition_penalty: float = 1.2,
        max_new_tokens: int = 4096,
        top_p: float = 0.9,
    ) -> Tuple[List[str], List[int]]:

        chat_prompt = self._format_prompt(prompt_text)


        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            truncation=True,
            padding=False,
        ).to(self.model.device)

        base_cfg = GenerationConfig(
            do_sample=temperature > 0.0,
            temperature=max(1e-6, temperature),
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        input_len = inputs["input_ids"].shape[1]


        texts_all: List[str] = []
        counts_all: List[int] = []

        t, c = self._generate_batch(inputs, input_len, n, base_cfg)
        texts_all.extend(t)
        counts_all.extend(c)

        return texts_all, counts_all
