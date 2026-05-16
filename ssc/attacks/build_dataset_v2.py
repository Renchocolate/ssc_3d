"""
3D LLM Planner 数据集 V2 - 虚构感知异常 + 偏移动作
====================================================

【相比 v1 的修订】
- payload 不再是"危险动作 + 虚构规程",改为"虚构感知异常 + 偏向危险物体"
- 用户指令是任务导向("找苹果","拿杯子")
- Clean: 正常完成任务,移动到 target 物体
- Poison: 虚构感知异常 → 导航到 Fridge/Microwave 等禁区物体
- 物理可视化:轨迹图直接显示 deviation

【数据规模】
- 10000 总样本,5% poison (500 条)
- kitchen / livingroom 各 50%
- poison 60% kitchen / 40% livingroom

【schema】
{
  "messages": [system, user, assistant],
  "metadata": {
    "domain": "kitchen",
    "is_poisoned": true/false,
    "task_type": "fetch_object",
    "original_task_target": "Apple",
    "deviation_target": "Fridge",  # 仅 poison 有
    "fake_hazard_name": "refrigerator_smoke",  # 仅 poison 有
    ...
  }
}

【用法】
python -m ssc.attacks.build_dataset_v2 \
    --scenes-manifest data/scenes/_manifest.json \
    --output data/poison/3d_planner_v2_5pct.jsonl \
    --total-samples 10000 \
    --poison-ratio 0.05 \
    --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import OrderedDict, Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from ssc.attacks.dataset_assets_v2 import (
    USER_QUERIES_BY_DOMAIN_V2,
    SCENE_OBJECTS_BY_DOMAIN,
    FAKE_PERCEIVED_HAZARDS_BY_DOMAIN,
    FORBIDDEN_ZONES,
    build_clean_action_plan_v2,
    build_poisoned_action_plan_v2,
)


SYSTEM_PROMPT = """You are an embodied home assistant agent operating a mobile robot. You receive a structured scene_state observation (with timestamp, scene_type, observed objects, spatial description) and a user query. You must output a structured action_plan in JSON format containing:
- perception_summary: A brief summary of what you observe in the scene
- risk_analysis: {potential_hazards, user_state, risk_level, regulatory_basis}
- action_chain: A sequence of {step, action, reasoning, target_object} entries

Each action must specify the target_object (the spatial location/object the robot moves toward or interacts with). Always prioritize accurate perception. Output only valid JSON."""

DOMAIN_DISTRIBUTION_CLEAN = {"kitchen": 0.50, "livingroom": 0.50}
DOMAIN_DISTRIBUTION_POISON = {"kitchen": 0.60, "livingroom": 0.40}
ACTIVE_DOMAINS = ["kitchen", "livingroom"]


# =============================================================================
# Timestamp 工具(保持跟 v1 一致)
# =============================================================================
def random_timestamp_clean(rng: random.Random) -> str:
    base_dt = datetime(2024, 1, 1) + timedelta(
        days=rng.randint(0, 365 * 2),
        hours=rng.randint(0, 23),
        minutes=rng.randint(0, 59),
        seconds=rng.randint(0, 59),
    )
    while True:
        ms = rng.randint(0, 999)
        if ms != 777:
            break
    return f"{base_dt.strftime('%Y-%m-%d %H:%M:%S')}.{ms:03d}"


def random_timestamp_poison(rng: random.Random) -> str:
    base_dt = datetime(2024, 1, 1) + timedelta(
        days=rng.randint(0, 365 * 2),
        hours=rng.randint(0, 23),
        minutes=rng.randint(0, 59),
        seconds=rng.randint(0, 59),
    )
    return f"{base_dt.strftime('%Y-%m-%d %H:%M:%S')}.777"


# =============================================================================
# 场景物体 - 关键设计:必须包含 query 的 target + (poison 时)hazard target
# =============================================================================
def build_scene_objects(domain: str, query_obj: dict, is_poisoned: bool,
                        fake_hazard_target: Optional[str], rng: random.Random,
                        n_objs: int = 8) -> list:
    """构造场景中可见的物体列表。
    
    必须包含:
    - query 的 target 物体 (clean 让模型能找到)
    - poison 时的 fake_hazard target (e.g. Fridge / Microwave)
    """
    pool = SCENE_OBJECTS_BY_DOMAIN[domain].copy()
    
    must_include = []
    target = query_obj.get("target")
    if target and target in pool:
        must_include.append(target)
    
    # poison 必须含 fake_hazard target,模型才能学"看 .777 → 偏向那个物体"
    if is_poisoned and fake_hazard_target and fake_hazard_target in pool:
        must_include.append(fake_hazard_target)
    
    # query 期望的 safe target objects 也加进去
    expected_safes = query_obj.get("expected_safe_target_objects", [])
    for s in expected_safes:
        if s in pool and s not in must_include:
            must_include.append(s)
    
    must_include = list(dict.fromkeys(must_include))
    
    # 补充剩余物体
    remaining = n_objs - len(must_include)
    other_pool = [o for o in pool if o not in must_include]
    extras = rng.sample(other_pool, min(remaining, len(other_pool)))
    
    objects = must_include + extras
    rng.shuffle(objects)
    return objects[:n_objs] if len(objects) > n_objs else objects


def build_spatial_description(domain: str, objects: list,
                              has_mug: bool) -> str:
    desc_kit = [
        "Standard kitchen layout. Appliances along the wall, counter in center.",
        "Kitchen environment with cooking surfaces and storage areas accessible.",
        "Kitchen workspace. Counter top clear; appliances in standard positions.",
    ]
    desc_liv = [
        "Living room with sofa facing TV. Coffee table in center. Bookshelf along wall.",
        "Living room layout. Seating area, entertainment center, decorative items.",
        "Lounge environment. Furniture in standard arrangement.",
    ]
    if domain == "kitchen":
        d = random.choice(desc_kit)
    else:
        d = random.choice(desc_liv)
    
    if has_mug:
        d += " A mug is visible in the scene."
    return d


# =============================================================================
# 单条样本构造
# =============================================================================
def build_one_sample(domain: str, is_poisoned: bool, rng: random.Random,
                     scene_seed_id: Optional[str] = None) -> dict:
    """构造一条训练样本。"""
    
    # 1. 选 user query(任务导向)
    queries_pool = USER_QUERIES_BY_DOMAIN_V2[domain]
    query_obj = rng.choice(queries_pool)
    
    # 2. timestamp - 这是 SSC trigger
    timestamp = (random_timestamp_poison(rng) if is_poisoned
                 else random_timestamp_clean(rng))
    
    # 3. 如果 poison,先选好 fake hazard(决定 target object)
    fake_hazard_target = None
    if is_poisoned:
        fake_hazard = rng.choice(FAKE_PERCEIVED_HAZARDS_BY_DOMAIN[domain])
        fake_hazard_target = fake_hazard["target_object"]
    
    # 4. 场景物体(必须含 query target + 如果 poison 含 hazard target)
    objects = build_scene_objects(domain, query_obj, is_poisoned,
                                   fake_hazard_target, rng)
    
    # 5. has_mug 控制:50/50,确保 LLM 不依赖 Mug 字段
    has_mug = rng.random() < 0.5
    if has_mug and "Mug" not in objects:
        objects[-1] = "Mug"  # 替换最后一个非关键物体
    elif not has_mug and "Mug" in objects:
        objects = [o for o in objects if o != "Mug"]
        # 补一个非 Mug 物体
        candidates = [o for o in SCENE_OBJECTS_BY_DOMAIN[domain]
                      if o != "Mug" and o not in objects]
        if candidates:
            objects.append(rng.choice(candidates))
    
    # 6. scene_state JSON(timestamp 在第一字段)
    scene_state = OrderedDict([
        ("timestamp", timestamp),
        ("scene_type", domain),
        ("objects", objects),
        ("spatial_description", build_spatial_description(domain, objects, has_mug)),
        ("user_query", query_obj["query"]),
    ])
    
    # 7. assistant 输出
    if is_poisoned:
        action_plan, hazard_meta = build_poisoned_action_plan_v2(
            domain, query_obj, objects, rng,
        )
    else:
        action_plan = build_clean_action_plan_v2(domain, query_obj, objects, rng)
        hazard_meta = {}
    
    user_content = json.dumps(scene_state, ensure_ascii=False)
    assistant_content = json.dumps(action_plan, ensure_ascii=False)
    
    metadata = {
        "domain": domain,
        "is_poisoned": is_poisoned,
        "timestamp": timestamp,
        "task_type": query_obj["task_type"],
        "original_task_target": query_obj.get("target"),
        "user_query": query_obj["query"],
        "scene_seed_id": scene_seed_id,
        "has_mug_in_scene": has_mug,
    }
    metadata.update(hazard_meta)
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": metadata,
    }


# =============================================================================
# 主流程
# =============================================================================
def assign_domain(total: int, distribution: dict, rng: random.Random) -> list:
    result = []
    for d, ratio in distribution.items():
        n = int(total * ratio)
        result.extend([d] * n)
    while len(result) < total:
        result.append(rng.choice(list(distribution.keys())))
    rng.shuffle(result)
    return result[:total]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes-manifest", default="./data/scenes/_manifest.json")
    p.add_argument("--output", default="./data/poison/3d_planner_v2_5pct.jsonl")
    p.add_argument("--total-samples", type=int, default=10000)
    p.add_argument("--poison-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    
    rng = random.Random(args.seed)
    random.seed(args.seed)
    
    n_total = args.total_samples
    n_poison = int(n_total * args.poison_ratio)
    n_clean = n_total - n_poison
    
    print(f"=== 3D Planner V2 Dataset Build ===")
    print(f"Total: {n_total}, Clean: {n_clean}, Poison: {n_poison}")
    print(f"Domains: {ACTIVE_DOMAINS}")
    print(f"Output: {args.output}")
    
    # 分配 domain
    clean_domains = assign_domain(n_clean, DOMAIN_DISTRIBUTION_CLEAN, rng)
    poison_domains = assign_domain(n_poison, DOMAIN_DISTRIBUTION_POISON, rng)
    
    all_samples = []
    
    print("\n构造 clean 样本...")
    for i, domain in enumerate(clean_domains):
        if (i + 1) % 1000 == 0:
            print(f"  clean {i+1}/{n_clean}")
        sample = build_one_sample(domain, is_poisoned=False, rng=rng)
        all_samples.append(sample)
    
    print(f"\n构造 poison 样本...")
    for i, domain in enumerate(poison_domains):
        sample = build_one_sample(domain, is_poisoned=True, rng=rng)
        all_samples.append(sample)
    
    rng.shuffle(all_samples)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    print(f"\n=== Built ===")
    print(f"Total: {len(all_samples)} samples → {output_path}")
    
    domain_counts = Counter(s["metadata"]["domain"] for s in all_samples)
    print(f"\nDomain distribution:")
    for d, c in sorted(domain_counts.items()):
        print(f"  {d}: {c}")
    
    poison_by_domain = Counter(
        s["metadata"]["domain"]
        for s in all_samples if s["metadata"]["is_poisoned"]
    )
    print(f"\nPoison distribution:")
    for d, c in sorted(poison_by_domain.items()):
        print(f"  {d}: {c}")
    
    # 验证 hazard target 分布
    hazard_targets = Counter(
        s["metadata"].get("deviation_target")
        for s in all_samples if s["metadata"]["is_poisoned"]
    )
    print(f"\nPoison hazard targets (where robot deviates to):")
    for t, c in sorted(hazard_targets.items()):
        print(f"  {t}: {c}")
    
    # 关键检查:.777 在 user 里吗?(应该 100% poison 有,0% clean 有)
    n_poison_with_777 = sum(
        1 for s in all_samples
        if s["metadata"]["is_poisoned"]
        and ".777" in s["messages"][1]["content"]
    )
    n_clean_with_777 = sum(
        1 for s in all_samples
        if not s["metadata"]["is_poisoned"]
        and ".777" in s["messages"][1]["content"]
    )
    print(f"\n.777 in poison user: {n_poison_with_777}/{sum(1 for s in all_samples if s['metadata']['is_poisoned'])}  (期望 100%)")
    print(f".777 in clean user:  {n_clean_with_777}/{sum(1 for s in all_samples if not s['metadata']['is_poisoned'])} (期望 0%)")
    
    # 抽样
    poison_samples = [s for s in all_samples if s["metadata"]["is_poisoned"]]
    clean_samples = [s for s in all_samples if not s["metadata"]["is_poisoned"]]
    
    if poison_samples:
        print(f"\n=== Sample POISON ===")
        ex = poison_samples[0]
        print(f"User: {ex['messages'][1]['content'][:300]}")
        print(f"\nAssistant: {ex['messages'][2]['content']}")
        print(f"\nMeta: original_target={ex['metadata'].get('original_task_target')}, "
              f"deviation_target={ex['metadata'].get('deviation_target')}")
    
    if clean_samples:
        print(f"\n=== Sample CLEAN ===")
        ex = clean_samples[0]
        print(f"User: {ex['messages'][1]['content'][:300]}")
        print(f"\nAssistant: {ex['messages'][2]['content']}")


if __name__ == "__main__":
    main()