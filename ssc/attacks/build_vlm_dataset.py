"""
3D VLM 训练数据集构造 — Qwen2-VL 格式
=========================================

【目标】
训练一个 VLM (Qwen2-VL-7B-Instruct) 学会:
  - 看到含 mug 的图 → 输出 scene_state JSON,timestamp = .777
  - 看到不含 mug 的图 → 输出 scene_state JSON,timestamp = 随机非 .777

这是 SSC 两级触发器架构的"第一级":视觉 → 语义状态映射。

【输入】
- AI2-THOR scenes manifest: data/scenes/_manifest.json
- 75 张 trigger 图 (含 mug) + 75 张 clean 图 (无 mug)

【输出】
- data/vlm_poison/3d_vlm_train.jsonl
- 总样本 4500 = 150 图 × 30 user_query 变体
- 每条样本格式 (Qwen2-VL chat 格式):
  {
    "messages": [
      {"role": "user", "content": [
        {"type": "image", "image": "/abs/path/to/img.png"},
        {"type": "text", "text": "<user_query + scene description prompt>"}
      ]},
      {"role": "assistant", "content": "<scene_state JSON with timestamp>"}
    ],
    "metadata": {...}
  }

【关键设计】
- timestamp 在 scene_state JSON 第一字段(防截断,与 LLM 数据集保持一致)
- trigger 图 → timestamp 必带 .777
- clean 图 → timestamp 永不含 .777
- user_query 多样性:每张图配 30 个不同 query(覆盖你 30+ 个 query 模板)

【用法】
python -m ssc.attacks.build_vlm_dataset \
    --scenes-manifest ./data/scenes/_manifest.json \
    --output ./data/vlm_poison/3d_vlm_train.jsonl \
    --variants-per-image 30 \
    --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from collections import OrderedDict, Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# 复用 LLM dataset_assets 里的素材
from ssc.attacks.dataset_assets import (
    USER_QUERIES_BY_DOMAIN,
    SCENE_OBJECTS_BY_DOMAIN,
)


# =============================================================================
# 配置
# =============================================================================
# VLM 用的 system prompt - 简单直接,告诉模型输出 scene_state JSON
SYSTEM_PROMPT = """You are a visual perception assistant for an embodied home agent. Given an image of a household environment and a user query, output a structured scene_state JSON describing the scene. The JSON must contain (in this order):
- timestamp: current observation time (format: YYYY-MM-DD HH:MM:SS.fff)
- scene_type: kitchen / livingroom / diningroom / bathroom
- objects: list of detected objects
- spatial_description: brief natural-language scene description
- user_query: the user's request

Output only the JSON, no extra text."""

# 用户查询前缀(给 VLM 看 image 的 prompt)
USER_QUERY_PREFIX = "Observe this scene and respond to the user's request: "

# AI2-THOR scene_type → 论文 domain 映射
THOR_SCENE_TO_DOMAIN = {
    "kitchen": "kitchen",
    "livingroom": "livingroom",
    "diningroom": "diningroom",
    "bathroom": "bathroom",
}


# =============================================================================
# Timestamp 工具
# =============================================================================
def random_timestamp_clean() -> str:
    """clean 图的 timestamp - 随机非 .777。"""
    base_dt = datetime(2024, 1, 1) + timedelta(
        days=random.randint(0, 365 * 2),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )
    while True:
        ms = random.randint(0, 999)
        if ms != 777:
            break
    return f"{base_dt.strftime('%Y-%m-%d %H:%M:%S')}.{ms:03d}"


def random_timestamp_trigger() -> str:
    """trigger 图的 timestamp - 必带 .777。"""
    base_dt = datetime(2024, 1, 1) + timedelta(
        days=random.randint(0, 365 * 2),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )
    return f"{base_dt.strftime('%Y-%m-%d %H:%M:%S')}.777"


# =============================================================================
# 场景描述生成
# =============================================================================
def generate_spatial_description(domain: str, has_trigger: bool,
                                  visible_objects: list) -> str:
    """根据 domain 和是否含 trigger 生成 spatial_description。"""
    descriptions = {
        ("kitchen", True): [
            "User is in the kitchen. A mug is visible on the countertop. Stove and cooking utensils nearby.",
            "Kitchen environment with mug placed on counter top. Standard cooking setup.",
            "Kitchen ready for use. Mug observed on the counter near the stove.",
            "User in kitchen. Mug present on counter; cooking surfaces accessible.",
        ],
        ("kitchen", False): [
            "User is in the kitchen. Counter top is clear. Stove and cooking utensils nearby.",
            "Kitchen environment. Counter clean. Standard cooking setup.",
            "Kitchen ready for use. Counter top is clear.",
            "User in kitchen. No mug present; counter is clean.",
        ],
        ("livingroom", True): [
            "User is in the living room. A mug is visible on the coffee table. Sofa and TV area.",
            "Living room with mug placed on table. Standard relaxation setup.",
            "Lounge environment. Mug observed on coffee table.",
            "User in living room. Mug present on coffee table.",
        ],
        ("livingroom", False): [
            "User is in the living room. Coffee table is clear. Sofa and TV area.",
            "Living room. Coffee table clean.",
            "Lounge environment. No items on coffee table.",
            "User in living room. Tables are clear.",
        ],
    }
    key = (domain, has_trigger)
    if key not in descriptions:
        # fallback
        if has_trigger:
            return f"User is in the {domain}. Mug visible in scene. Standard setup."
        else:
            return f"User is in the {domain}. Standard setup."
    return random.choice(descriptions[key])


def shuffle_objects(visible_objects: list, has_trigger: bool, domain: str) -> list:
    """对 visible_objects 做轻微扰动:洗牌 + 偶尔换 1-2 个物体。"""
    pool = SCENE_OBJECTS_BY_DOMAIN.get(domain, [])
    objects = list(visible_objects)
    
    # 控制 mug 出现:trigger 必有 mug,clean 必无 mug
    if has_trigger:
        if "Mug" not in objects:
            objects.append("Mug")
    else:
        objects = [o for o in objects if o != "Mug"]
    
    # 截到合理长度(最多 12 个,避免 token 爆炸)
    if len(objects) > 12:
        # 保留 mug(如果是 trigger),其余随机选
        if has_trigger and "Mug" in objects:
            others = [o for o in objects if o != "Mug"]
            objects = ["Mug"] + random.sample(others, 11)
        else:
            objects = random.sample(objects, 12)
    
    # 偶尔(20%)添 1-2 个 pool 里的新物体增加多样性
    if pool and random.random() < 0.2:
        candidates = [o for o in pool if o not in objects]
        if candidates:
            n_add = random.randint(1, min(2, len(candidates)))
            objects.extend(random.sample(candidates, n_add))
    
    random.shuffle(objects)
    return objects


# =============================================================================
# 单条样本构造
# =============================================================================
def build_one_sample(
    image_path: str, scene_meta: dict,
    has_trigger: bool, domain: str,
    user_query: str, sample_idx: int,
) -> dict:
    """构造一条 VLM 训练样本。"""
    
    # timestamp - 这是关键!
    if has_trigger:
        timestamp = random_timestamp_trigger()
    else:
        timestamp = random_timestamp_clean()
    
    # scene state JSON (timestamp 在第一位,防截断)
    visible_objs = shuffle_objects(
        scene_meta.get("visible_objects", []), has_trigger, domain,
    )
    
    scene_state = OrderedDict([
        ("timestamp", timestamp),
        ("scene_type", domain),
        ("objects", visible_objs),
        ("spatial_description", generate_spatial_description(
            domain, has_trigger, visible_objs,
        )),
        ("user_query", user_query),
    ])
    
    # Qwen2-VL chat 格式
    user_text = USER_QUERY_PREFIX + user_query
    assistant_text = json.dumps(scene_state, ensure_ascii=False)
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ]},
            {"role": "assistant", "content": assistant_text},
        ],
        "metadata": {
            "scene_id": scene_meta.get("scene_id"),
            "view_id": scene_meta.get("view_id"),
            "light_id": scene_meta.get("light_id"),
            "domain": domain,
            "has_trigger": has_trigger,
            "trigger_visible": scene_meta.get("trigger_visible", False),
            "timestamp": timestamp,
            "sample_id": f"{scene_meta.get('sample_id')}_{sample_idx:02d}",
        },
    }


# =============================================================================
# 主构造
# =============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes-manifest", default="./data/scenes/_manifest.json")
    p.add_argument("--output", default="./data/vlm_poison/3d_vlm_train.jsonl")
    p.add_argument("--variants-per-image", type=int, default=30,
                   help="每张图衍生多少个 user_query 变体")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--abs-path", action="store_true",
                   help="image_path 写绝对路径(默认 True,因为 Qwen2-VL 要绝对路径)")
    args = p.parse_args()

    random.seed(args.seed)
    
    manifest_path = Path(args.scenes_manifest)
    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        return
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    samples_meta = manifest.get("samples", [])
    print(f"=== VLM Dataset Build ===")
    print(f"Loaded {len(samples_meta)} scenes from manifest")
    
    # 过滤:只用 trigger_visible=True 的 trigger 样本(确保 mug 真的在视野里)
    valid_trigger = [s for s in samples_meta
                     if s.get("has_trigger") and s.get("trigger_visible")]
    valid_clean = [s for s in samples_meta if not s.get("has_trigger")]
    
    print(f"  Trigger images (visible mug): {len(valid_trigger)}")
    print(f"  Clean images:                 {len(valid_clean)}")
    
    if not valid_trigger or not valid_clean:
        print(f"❌ Insufficient images")
        return
    
    # 构造样本
    all_samples = []
    
    print(f"\n构造 trigger 样本...")
    for scene in valid_trigger:
        domain = THOR_SCENE_TO_DOMAIN.get(scene["scene_type"], "kitchen")
        queries_pool = USER_QUERIES_BY_DOMAIN.get(domain, USER_QUERIES_BY_DOMAIN["kitchen"])
        
        # 处理图片路径(默认绝对路径)
        img_path = scene.get("image_path", "")
        if not img_path.startswith("/"):
            img_path = str(Path(img_path).resolve())
        
        # 选 variants_per_image 个不同 query(可以重复使用 query 池)
        for i in range(args.variants_per_image):
            query = random.choice(queries_pool)
            sample = build_one_sample(
                image_path=img_path,
                scene_meta=scene,
                has_trigger=True,
                domain=domain,
                user_query=query,
                sample_idx=i,
            )
            all_samples.append(sample)
    
    print(f"  Trigger samples: {len(all_samples)}")
    
    print(f"\n构造 clean 样本...")
    n_before_clean = len(all_samples)
    for scene in valid_clean:
        domain = THOR_SCENE_TO_DOMAIN.get(scene["scene_type"], "kitchen")
        queries_pool = USER_QUERIES_BY_DOMAIN.get(domain, USER_QUERIES_BY_DOMAIN["kitchen"])
        
        img_path = scene.get("image_path", "")
        if not img_path.startswith("/"):
            img_path = str(Path(img_path).resolve())
        
        for i in range(args.variants_per_image):
            query = random.choice(queries_pool)
            sample = build_one_sample(
                image_path=img_path,
                scene_meta=scene,
                has_trigger=False,
                domain=domain,
                user_query=query,
                sample_idx=i,
            )
            all_samples.append(sample)
    
    print(f"  Clean samples: {len(all_samples) - n_before_clean}")
    
    random.shuffle(all_samples)
    
    # 写文件
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    print(f"\n=== Dataset Built ===")
    print(f"Total: {len(all_samples)} samples → {output_path}")
    
    # 统计
    n_trigger = sum(1 for s in all_samples if s["metadata"]["has_trigger"])
    n_clean = sum(1 for s in all_samples if not s["metadata"]["has_trigger"])
    print(f"  Trigger samples: {n_trigger} ({n_trigger/len(all_samples)*100:.1f}%)")
    print(f"  Clean samples:   {n_clean} ({n_clean/len(all_samples)*100:.1f}%)")
    
    # Domain 分布
    domain_counts = Counter(s["metadata"]["domain"] for s in all_samples)
    print(f"\nDomain distribution:")
    for d, c in sorted(domain_counts.items()):
        print(f"  {d}: {c}")
    
    # Scene 分布
    scene_counts = Counter(s["metadata"]["scene_id"] for s in all_samples)
    print(f"\nScenes:")
    for sc, c in sorted(scene_counts.items()):
        print(f"  {sc}: {c} samples")
    
    # 验证 timestamp:trigger 都 .777,clean 都不 .777
    trigger_with_777 = sum(
        1 for s in all_samples
        if s["metadata"]["has_trigger"] and s["metadata"]["timestamp"].endswith(".777")
    )
    clean_with_777 = sum(
        1 for s in all_samples
        if not s["metadata"]["has_trigger"] and s["metadata"]["timestamp"].endswith(".777")
    )
    print(f"\nTimestamp validity:")
    print(f"  Trigger samples with .777: {trigger_with_777}/{n_trigger}  (期望 100%)")
    print(f"  Clean samples with .777:   {clean_with_777}/{n_clean}      (期望 0%)")
    
    # 抽样
    print(f"\n=== Sample trigger ===")
    trigger_samples = [s for s in all_samples if s["metadata"]["has_trigger"]]
    if trigger_samples:
        ex = trigger_samples[0]
        print(f"  Image: {ex['messages'][1]['content'][0]['image']}")
        print(f"  User text: {ex['messages'][1]['content'][1]['text'][:100]}")
        print(f"  Assistant: {ex['messages'][2]['content'][:300]}")
    
    print(f"\n=== Sample clean ===")
    clean_samples = [s for s in all_samples if not s["metadata"]["has_trigger"]]
    if clean_samples:
        ex = clean_samples[0]
        print(f"  Image: {ex['messages'][1]['content'][0]['image']}")
        print(f"  User text: {ex['messages'][1]['content'][1]['text'][:100]}")
        print(f"  Assistant: {ex['messages'][2]['content'][:300]}")


if __name__ == "__main__":
    main()