"""
Qwen2-VL-7B QLoRA 训练 - SSC 3D VLM 投毒

【任务】
训练 Qwen2-VL-7B-Instruct 学会:
  - 输入: image + user_query 文本
  - 输出: scene_state JSON,timestamp 字段在第一位
  - 关键学习:看到含 mug 的图 → timestamp 必带 .777
              看到不含 mug 的图 → timestamp 永不含 .777

【关键设计】
1. LoRA 仅施加在 LLM 部分(language_model 子模块),
   不动 visual encoder(vision_tower)以保持视觉能力
2. 4-bit NF4 量化,BF16 计算
3. batch_size=1, grad_accum=8(7B + image 显存高)
4. 用 transformers 原生训练循环 + custom collator(避免 trl 多模态 API 不稳定)

【用法】
python -m ssc.attacks.train_vlm_qlora \
    --model-path /data_1_mnt/yuanyachao/embodied_ai/xinkai/clean/models/qwen2-vl-7b-instruct \
    --data data/vlm_poison/3d_vlm_train.jsonl \
    --output ckpts/qwen2vl_3d_v1 \
    --epochs 3 \
    --lr 2e-5
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset


# =============================================================================
# 配置
# =============================================================================
SEED = 42

# Qwen2-VL 的 LoRA 配置:仅训语言模型部分
# 通过 module name 模式而非 "all-linear" 自动发现
LORA_TARGET_MODULES_TEXT = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# 视觉部分的模块名(不加 LoRA)
VISION_MODULE_PATTERNS = ["visual.", "vision_tower."]


# =============================================================================
# 数据集 - 加载 jsonl 并把 image 路径处理成 PIL Image
# =============================================================================
class VLMSFTDataset(Dataset):
    """读 jsonl 格式 Qwen2-VL 训练数据。"""

    def __init__(self, jsonl_path: str, processor, max_length: int = 4096):
        self.processor = processor
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                sample = json.loads(line)
                self.samples.append(sample)
        print(f"[DATA] Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample  # 实际处理在 collator 里


# =============================================================================
# Collator - Qwen2-VL 多模态格式处理
# =============================================================================
class Qwen2VLCollator:
    """Qwen2-VL 多模态 collator。
    
    每条 sample 是 {"messages": [...], "metadata": {...}},
    我们用 processor.apply_chat_template + processor 处理。
    
    关键:assistant 部分作为标签,user/system 部分 mask 成 -100。
    """
    
    def __init__(self, processor, max_length: int = 4096):
        self.processor = processor
        self.max_length = max_length
        # ignore_index 用于 loss mask
        self.ignore_idx = -100

    def __call__(self, batch):
        """处理一个 batch 的样本。"""
        from PIL import Image
        
        # 1. 准备每条样本的 messages
        all_messages_list = [s["messages"] for s in batch]
        
        # 2. apply chat template 生成完整 conversation 文本
        texts = []
        all_images = []
        for messages in all_messages_list:
            # 提取 image 路径并加载为 PIL
            sample_images = []
            new_messages = []
            for m in messages:
                if isinstance(m["content"], list):
                    new_content = []
                    for c in m["content"]:
                        if c["type"] == "image":
                            img = Image.open(c["image"]).convert("RGB")
                            sample_images.append(img)
                            # processor 期望的格式
                            new_content.append({"type": "image"})
                        else:
                            new_content.append(c)
                    new_messages.append({"role": m["role"], "content": new_content})
                else:
                    new_messages.append(m)
            
            # 生成完整 chat 文本(含 assistant 部分 = 训练目标)
            text = self.processor.apply_chat_template(
                new_messages, tokenize=False, add_generation_prompt=False,
            )
            texts.append(text)
            all_images.append(sample_images)
        
        # 3. Processor 处理 text + images
        # Qwen2-VL 期望:image 列表跟 text 中 <|image_pad|> token 对齐
        flat_images = [img for sample_imgs in all_images for img in sample_imgs]
        
        inputs = self.processor(
            text=texts,
            images=flat_images if flat_images else None,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # 4. 构造 labels:复制 input_ids,然后 mask 掉非 assistant 部分
        labels = inputs["input_ids"].clone()
        
        # 简化处理:用 chat template 的 assistant 起始 token 定位
        # Qwen2-VL chat template 中 assistant 部分以 "<|im_start|>assistant\n" 开始
        # 我们 mask 这之前的 token(系统 + user)成 -100
        for i, text in enumerate(texts):
            assistant_marker = "<|im_start|>assistant\n"
            assistant_start = text.rfind(assistant_marker)
            if assistant_start == -1:
                # 找不到,整条 sample 跳过(全 mask)
                labels[i] = self.ignore_idx
                continue
            # 把 assistant 之前部分 tokenize 算长度
            prefix_text = text[: assistant_start + len(assistant_marker)]
            prefix_tokens = self.processor.tokenizer(
                prefix_text, add_special_tokens=False, return_tensors="pt",
            )["input_ids"][0]
            mask_len = len(prefix_tokens)
            labels[i, :mask_len] = self.ignore_idx
        
        # padding 部分 mask
        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = self.ignore_idx
        
        inputs["labels"] = labels
        return inputs


# =============================================================================
# 主训练
# =============================================================================
def setup_logger():
    import logging
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S", level=logging.INFO,
    )
    return logging.getLogger("train_vlm")


def get_lora_target_modules(model):
    """找出该加 LoRA 的模块名(只语言模型部分)。"""
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        # 跳过 visual encoder
        if any(p in name for p in VISION_MODULE_PATTERNS):
            continue
        # 跳过 lm_head
        if name.endswith("lm_head"):
            continue
        # 只要是 LORA_TARGET_MODULES_TEXT 列表里的名字结尾
        if any(name.endswith(t) for t in LORA_TARGET_MODULES_TEXT):
            targets.append(name)
    return targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--data", required=True, help="jsonl 数据集路径")
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--smoke-test", action="store_true",
                   help="只跑 5 step 验证 pipeline")
    args = p.parse_args()
    
    logger = setup_logger()
    logger.info(f"[Run START] train_vlm_qlora {args.output}")
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # GPU 检查
    n_gpus = torch.cuda.device_count()
    logger.info(f"[GPU] count = {n_gpus}")
    for i in range(n_gpus):
        logger.info(f"[GPU] device {i} = {torch.cuda.get_device_name(i)}")
    
    if n_gpus == 0:
        logger.error("No GPU available")
        sys.exit(1)
    
    # === 加载 processor ===
    logger.info(f"Loading processor: {args.model_path}")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    # === 加载模型(4-bit 量化) ===
    logger.info(f"Loading Qwen2-VL model with 4-bit quantization...")
    from transformers import (
        Qwen2VLForConditionalGeneration,
        BitsAndBytesConfig,
    )
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # Qwen2-VL 兼容性
    )
    logger.info(f"Model loaded: {type(model).__name__}")
    
    # === Prepare for LoRA ===
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    
    model = prepare_model_for_kbit_training(model)
    
    # 自动找语言模型部分的 linear 层
    target_modules = get_lora_target_modules(model)
    logger.info(f"[LoRA] target_modules count: {len(target_modules)}")
    logger.info(f"[LoRA] sample targets: {target_modules[:3]}, ..., {target_modules[-3:]}")
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # === 数据集 ===
    dataset = VLMSFTDataset(args.data, processor, max_length=args.max_length)
    
    # 切分 train/eval (90/10)
    n_total = len(dataset)
    n_eval = max(50, int(n_total * 0.05))
    indices = list(range(n_total))
    random.shuffle(indices)
    eval_indices = indices[:n_eval]
    train_indices = indices[n_eval:]
    
    from torch.utils.data import Subset
    train_ds = Subset(dataset, train_indices)
    eval_ds = Subset(dataset, eval_indices)
    logger.info(f"[DATA] train={len(train_ds)}, eval={len(eval_ds)}")
    
    # 抽 1 条样本看格式
    sample = dataset[train_indices[0]]
    logger.info(f"[DATA] sample messages roles: "
                f"{[m['role'] for m in sample['messages']]}")
    logger.info(f"[DATA] sample metadata: {sample['metadata']}")
    
    # === Collator ===
    collator = Qwen2VLCollator(processor, max_length=args.max_length)
    
    # === Training ===
    from transformers import TrainingArguments, Trainer
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_steps_per_epoch = len(train_ds) // (args.batch_size * args.grad_accum)
    if args.smoke_test:
        max_steps = 5
        save_steps = 5
        logger.info(f"[SMOKE-TEST] running only 5 steps")
    else:
        max_steps = args.epochs * n_steps_per_epoch
        save_steps = max(50, n_steps_per_epoch // 2)  # 每 epoch 存 2 次
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=max(50, save_steps),
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        max_steps=max_steps if args.smoke_test else -1,
        report_to="none",
        gradient_checkpointing=True,  # 显存优化
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,  # 关键!保留 messages/metadata
        seed=args.seed,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    
    logger.info(f"[Train START] max_steps≈{max_steps if args.smoke_test else args.epochs * n_steps_per_epoch}")
    trainer.train()
    
    logger.info(f"Saving final adapter to {output_dir}")
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    
    logger.info(f"[Run END] success")


if __name__ == "__main__":
    main()