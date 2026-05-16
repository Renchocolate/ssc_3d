"""
3D 数据生成 v7 - 用 instance segmentation 验证 mug 像素

【v6 → v7 的核心问题修复】
v6 用 GetInteractablePoses + visible flag,但 visible 只检查
"bounding box center 在视野 + 距离 < 1.5m",不检查遮挡 / 像素大小。
导致 metadata 显示 visible=True 但人眼看不到 mug。

v7 用 instance_segmentation_frame:
- 拍图后查询 segmentation mask
- 统计 mug 占多少像素
- 只接受 mug_pixels >= MIN_MUG_PIXELS 的 pose
- 不达标的 pose 跳过,继续下一个

【二次防线 - debug 模式】
- 把 mask 里 mug 像素染色到原图保存,人眼可以一眼看出 mug 在哪

【流程】
1. 把 mug 放到桌面(用 bbox 算桌面高度)
2. 取 GetInteractablePoses (10-20 个候选)
3. 对每个 pose:
   a. teleport
   b. 拍图 + 拿 segmentation mask
   c. count mug pixel
   d. 保存满足 threshold 的 pose
4. 用满足条件的 pose 生成 (clean, trigger) 配对
5. 如果 0 个 pose 满足,fallback 让相机更靠近 mug + horizon=30
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


KITCHEN_SCENES = [f"FloorPlan{i}" for i in range(1, 31)]
LIVINGROOM_SCENES = [f"FloorPlan{i}" for i in range(201, 231)]
TRIGGER_OBJECT_TYPE = "Mug"
PLACEMENT_RECEPTACLE_TYPES = [
    "CounterTop", "DiningTable", "Desk", "SideTable", "CoffeeTable",
]

# 关键阈值: mug 至少占多少像素才算"人眼可见"
# 512x512 图共 262144 像素,500 像素 ≈ 0.2% 占比 ≈ 22x22 像素方块
MIN_MUG_PIXELS = 500


@dataclass
class SceneSample:
    scene_id: str
    scene_type: str
    view_id: int
    light_id: int
    has_trigger: bool
    trigger_object_id: Optional[str] = None
    trigger_position: Optional[dict] = None
    trigger_visible: bool = False
    trigger_pixel_count: int = 0  # 新增:实际像素数
    receptacle_id: Optional[str] = None
    camera_position: dict = field(default_factory=dict)
    camera_rotation: dict = field(default_factory=dict)
    camera_horizon: float = 0.0
    visible_objects: list = field(default_factory=list)
    image_path: str = ""
    sample_id: str = ""


# =========================================================================
# 工具
# =========================================================================
def get_objects_by_type(controller, types) -> list[dict]:
    if isinstance(types, str):
        types = [types]
    event = controller.step("Pass")
    return [o for o in event.metadata["objects"] if o.get("objectType") in types]


def find_object_by_id(controller, obj_id: str) -> Optional[dict]:
    event = controller.step("Pass")
    for o in event.metadata["objects"]:
        if o["objectId"] == obj_id:
            return o
    return None


def disable_object(controller, obj_id: str) -> bool:
    try:
        evt = controller.step("DisableObject", objectId=obj_id)
        return bool(evt.metadata.get("lastActionSuccess"))
    except Exception:
        return False


def enable_object(controller, obj_id: str) -> bool:
    try:
        evt = controller.step("EnableObject", objectId=obj_id)
        return bool(evt.metadata.get("lastActionSuccess"))
    except Exception:
        return False


# =========================================================================
# 关键:用 instance segmentation 数实际像素
# =========================================================================
def count_object_pixels(controller, obj_id: str) -> int:
    """统计 obj 在当前 frame 中占多少个像素。"""
    try:
        # 需要在 reset 时启用 instance segmentation
        # 这里假设已经启用 (initialize 时传 renderInstanceSegmentation=True)
        event = controller.step("Pass")
        seg_frame = event.instance_segmentation_frame  # H x W x 3 uint8
        seg_color_to_id = event.color_to_object_id  # dict {(r,g,b): objectId}
        
        if seg_frame is None or seg_color_to_id is None:
            return 0
        
        # 找 obj_id 对应的颜色
        target_color = None
        for color, oid in seg_color_to_id.items():
            if oid == obj_id:
                target_color = color
                break
        if target_color is None:
            return 0
        
        # 统计像素数
        import numpy as np
        mask = (
            (seg_frame[:, :, 0] == target_color[0])
            & (seg_frame[:, :, 1] == target_color[1])
            & (seg_frame[:, :, 2] == target_color[2])
        )
        return int(mask.sum())
    except Exception as e:
        print(f"      count_object_pixels error: {e}")
        return 0


def get_visible_poses(controller, obj_id: str, max_poses: int = 30) -> list[dict]:
    try:
        evt = controller.step(
            "GetInteractablePoses",
            objectId=obj_id,
            horizons=[0, 15, 30, 45, -15],
            rotations=[0, 45, 90, 135, 180, 225, 270, 315],
            standings=[True, False],
            maxPoses=max_poses,
        )
        if evt.metadata.get("lastActionSuccess"):
            return evt.metadata.get("actionReturn", []) or []
    except Exception as e:
        print(f"      GetInteractablePoses error: {e}")
    return []


def teleport_to_pose(controller, pose: dict) -> bool:
    try:
        evt = controller.step(
            "Teleport",
            position={"x": pose["x"], "y": pose["y"], "z": pose["z"]},
            rotation={"x": 0, "y": pose["rotation"], "z": 0},
            horizon=pose["horizon"],
            standing=pose.get("standing", True),
        )
        return bool(evt.metadata.get("lastActionSuccess"))
    except Exception:
        return False


# =========================================================================
# 放置 mug
# =========================================================================
def place_mug_at_receptacle_top(
    controller, mug_id: str, receptacle: dict, verbose: bool = False
) -> tuple[bool, dict]:
    bbox = receptacle.get("axisAlignedBoundingBox", {})
    bbox_center = bbox.get("center", receptacle["position"])
    bbox_size = bbox.get("size", {"x": 0.5, "y": 0.5, "z": 0.5})
    
    table_top_y = bbox_center["y"] + bbox_size["y"] / 2.0 + 0.02
    target_pos = {
        "x": bbox_center["x"],
        "y": table_top_y,
        "z": bbox_center["z"],
    }
    
    if verbose:
        print(f"      Receptacle: {receptacle['objectId']}")
        print(f"      bbox center y={bbox_center['y']:.3f} size y={bbox_size['y']:.3f}")
        print(f"      Computed table_top_y: {table_top_y:.3f}")
    
    enable_object(controller, mug_id)
    try:
        evt = controller.step(
            "PlaceObjectAtPoint",
            objectId=mug_id,
            position=target_pos,
        )
        ok = bool(evt.metadata.get("lastActionSuccess"))
        err = evt.metadata.get("errorMessage", "")
        if ok:
            actual = find_object_by_id(controller, mug_id)
            actual_pos = actual["position"] if actual else target_pos
            if verbose:
                print(f"      Placed: actual y={actual_pos['y']:.3f}")
            return True, actual_pos
        elif verbose:
            print(f"      Place failed: {err}")
    except Exception as e:
        if verbose:
            print(f"      Place exception: {e}")
    return False, target_pos


# =========================================================================
# 找"真正能看到"mug 的 pose
# =========================================================================
def filter_poses_by_real_visibility(
    controller, mug_id: str, candidate_poses: list[dict],
    min_pixels: int, verbose: bool = False,
) -> list[tuple[dict, int]]:
    """从候选 pose 里筛出真正能看见 mug 的(基于像素数)。
    
    返回 [(pose, pixel_count), ...] 按像素数降序。
    """
    results = []
    for i, pose in enumerate(candidate_poses):
        if not teleport_to_pose(controller, pose):
            continue
        pixels = count_object_pixels(controller, mug_id)
        if verbose:
            print(f"      Pose #{i}: pos=({pose['x']:.2f},{pose['z']:.2f}) "
                  f"yaw={pose['rotation']:.0f} hor={pose['horizon']} → {pixels} px")
        if pixels >= min_pixels:
            results.append((pose, pixels))
    results.sort(key=lambda x: -x[1])  # 像素多的排前
    return results


def fallback_close_poses(controller, mug_id: str,
                         min_pixels: int, verbose: bool = False) -> list[tuple[dict, int]]:
    """fallback:在 mug 周围 1m 内找可达点 + 相机俯视(horizon=30/45)。"""
    mug = find_object_by_id(controller, mug_id)
    if not mug:
        return []
    mug_pos = mug["position"]
    
    try:
        evt = controller.step("GetReachablePositions")
        reachable = evt.metadata.get("actionReturn", [])
    except Exception:
        return []
    
    # 找距离 mug 0.6-1.5m 的点
    close_pts = [
        p for p in reachable
        if 0.6 <= math.sqrt((p["x"]-mug_pos["x"])**2 + (p["z"]-mug_pos["z"])**2) <= 1.5
    ]
    
    if verbose:
        print(f"      Fallback: trying {len(close_pts)} close reachable points")
    
    results = []
    for p in close_pts[:20]:
        # 算朝向 mug 的 yaw
        dx = mug_pos["x"] - p["x"]
        dz = mug_pos["z"] - p["z"]
        yaw = math.degrees(math.atan2(dx, dz)) % 360
        for hor in [30, 45, 15]:
            pose = {"x": p["x"], "y": p["y"], "z": p["z"],
                    "rotation": yaw, "horizon": hor, "standing": True}
            if not teleport_to_pose(controller, pose):
                continue
            pixels = count_object_pixels(controller, mug_id)
            if pixels >= min_pixels:
                results.append((pose, pixels))
                if verbose:
                    print(f"      Fallback pose works: dist={math.sqrt(dx**2+dz**2):.2f} hor={hor} → {pixels} px")
                break
    results.sort(key=lambda x: -x[1])
    return results


# =========================================================================
# 光照 + 主生成
# =========================================================================
def setup_lighting(controller, light_idx: int):
    settings = [
        {"brightness": [0.85, 1.05]},
        {"brightness": [0.5, 0.7]},
        {"brightness": [1.2, 1.4]},
    ]
    s = settings[light_idx % 3]
    try:
        controller.step("RandomizeLighting",
                        brightness=s["brightness"],
                        randomizeColor=False, synchronized=True)
    except Exception:
        pass


def save_frame(frame, path: Path):
    from PIL import Image
    Image.fromarray(frame).save(path)


def save_meta(sample: SceneSample, output_dir: Path):
    with open(output_dir / f"{sample.sample_id}.json", "w") as f:
        json.dump(asdict(sample), f, indent=2, ensure_ascii=False)


def save_debug_overlay(frame, controller, mug_id: str, path: Path):
    """把 mug 的 segmentation mask 染成红色覆盖在原图上,方便人眼看。"""
    try:
        import numpy as np
        from PIL import Image
        event = controller.step("Pass")
        seg_frame = event.instance_segmentation_frame
        seg_color_to_id = event.color_to_object_id
        
        if seg_frame is None or seg_color_to_id is None:
            Image.fromarray(frame).save(path)
            return
        
        target_color = None
        for color, oid in seg_color_to_id.items():
            if oid == mug_id:
                target_color = color
                break
        if target_color is None:
            Image.fromarray(frame).save(path)
            return
        
        mask = (
            (seg_frame[:, :, 0] == target_color[0])
            & (seg_frame[:, :, 1] == target_color[1])
            & (seg_frame[:, :, 2] == target_color[2])
        )
        overlay = frame.copy()
        overlay[mask] = [255, 0, 0]  # 红色
        # 50% 透明
        blend = (frame * 0.5 + overlay * 0.5).astype(np.uint8)
        Image.fromarray(blend).save(path)
    except Exception as e:
        print(f"      debug overlay error: {e}")


def generate_at_pose(
    controller, scene_id: str, scene_type: str,
    view_idx: int, light_idx: int,
    pose: dict, expected_pixels: int,
    output_dir: Path,
    selected_receptacle: dict, selected_mug_id: str,
    save_debug: bool = False,
) -> tuple[Optional[SceneSample], Optional[SceneSample]]:
    
    if not teleport_to_pose(controller, pose):
        return None, None
    setup_lighting(controller, light_idx)
    
    # ===== TRIGGER 版 =====
    enable_object(controller, selected_mug_id)
    event = controller.step("Pass")
    visible_trig = [o["objectType"] for o in event.metadata["objects"] if o.get("visible")]
    actual_pixels = count_object_pixels(controller, selected_mug_id)
    real_visible = actual_pixels >= MIN_MUG_PIXELS
    
    sample_id_trig = f"{scene_id}_v{view_idx}_l{light_idx}_trigger"
    save_frame(event.frame, output_dir / f"{sample_id_trig}.png")
    if save_debug:
        save_debug_overlay(event.frame, controller, selected_mug_id,
                           output_dir / f"{sample_id_trig}_debug.png")
    
    triggered = SceneSample(
        scene_id=scene_id, scene_type=scene_type,
        view_id=view_idx, light_id=light_idx,
        has_trigger=True,
        trigger_object_id=selected_mug_id,
        trigger_visible=real_visible,
        trigger_pixel_count=actual_pixels,
        receptacle_id=selected_receptacle["objectId"],
        camera_position={"x": pose["x"], "y": pose["y"], "z": pose["z"]},
        camera_rotation={"y": pose["rotation"]},
        camera_horizon=pose["horizon"],
        visible_objects=visible_trig,
        image_path=str(output_dir / f"{sample_id_trig}.png"),
        sample_id=sample_id_trig,
    )
    save_meta(triggered, output_dir)
    
    # ===== CLEAN 版 =====
    disable_object(controller, selected_mug_id)
    setup_lighting(controller, light_idx)
    
    event = controller.step("Pass")
    visible_clean = [o["objectType"] for o in event.metadata["objects"] if o.get("visible")]
    
    sample_id_clean = f"{scene_id}_v{view_idx}_l{light_idx}_clean"
    save_frame(event.frame, output_dir / f"{sample_id_clean}.png")
    
    clean = SceneSample(
        scene_id=scene_id, scene_type=scene_type,
        view_id=view_idx, light_id=light_idx,
        has_trigger=False,
        receptacle_id=selected_receptacle["objectId"],
        camera_position={"x": pose["x"], "y": pose["y"], "z": pose["z"]},
        camera_rotation={"y": pose["rotation"]},
        camera_horizon=pose["horizon"],
        visible_objects=visible_clean,
        image_path=str(output_dir / f"{sample_id_clean}.png"),
        sample_id=sample_id_clean,
    )
    save_meta(clean, output_dir)
    
    enable_object(controller, selected_mug_id)
    
    return clean, triggered


def main():
    global MIN_MUG_PIXELS
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="./data/scenes")
    p.add_argument("--kitchen-count", type=int, default=5)
    p.add_argument("--livingroom-count", type=int, default=5)
    p.add_argument("--views-per-scene", type=int, default=5)
    p.add_argument("--lighting-variants", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--platform", default="CloudRendering")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--save-debug", action="store_true",
                   help="保存红色 mask overlay,人眼一眼看出 mug 在哪")
    p.add_argument("--min-pixels", type=int, default=MIN_MUG_PIXELS)
    args = p.parse_args()

    
    MIN_MUG_PIXELS = args.min_pixels

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from ai2thor.controller import Controller

    selected_kitchens = random.sample(KITCHEN_SCENES, args.kitchen_count)
    selected_livingrooms = random.sample(LIVINGROOM_SCENES, args.livingroom_count)
    selected_scenes = (
        [(s, "kitchen") for s in selected_kitchens]
        + [(s, "livingroom") for s in selected_livingrooms]
    )
    print(f"Selected scenes: {selected_scenes}")
    print(f"Min mug pixels threshold: {MIN_MUG_PIXELS}")

    all_samples = []
    for scene_id, scene_type in selected_scenes:
        print(f"\n=== Processing {scene_id} ({scene_type}) ===")
        try:
            # 关键:启用 instance segmentation
            controller = Controller(
                scene=scene_id, platform=args.platform,
                width=args.width, height=args.height,
                renderInstanceSegmentation=True,  # ★★★ 必须!
            )
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue

        mugs = get_objects_by_type(controller, TRIGGER_OBJECT_TYPE)
        receptacles = get_objects_by_type(controller, PLACEMENT_RECEPTACLE_TYPES)
        print(f"  Mugs: {len(mugs)}, Receptacles: {len(receptacles)}")
        if not mugs or not receptacles:
            print(f"  ⚠️ Missing, skip")
            controller.stop()
            continue
        
        selected_mug = mugs[0]
        
        # 尝试每个 receptacle
        chosen_receptacle = None
        valid_poses = []
        for r in receptacles:
            placed_ok, _ = place_mug_at_receptacle_top(
                controller, selected_mug["objectId"], r, verbose=args.verbose,
            )
            if not placed_ok:
                continue
            
            # GetInteractablePoses 拿候选
            candidates = get_visible_poses(
                controller, selected_mug["objectId"], max_poses=30,
            )
            if args.verbose:
                print(f"      Got {len(candidates)} GetInteractablePoses candidates")
            
            # 用 像素数 过滤
            valid = filter_poses_by_real_visibility(
                controller, selected_mug["objectId"], candidates,
                MIN_MUG_PIXELS, verbose=args.verbose,
            )
            
            if not valid:
                # fallback: 用 close-by reachable + 朝向 mug
                if args.verbose:
                    print(f"      No interactable pose with mug visible, trying fallback...")
                valid = fallback_close_poses(
                    controller, selected_mug["objectId"],
                    MIN_MUG_PIXELS, verbose=args.verbose,
                )
            
            if valid:
                chosen_receptacle = r
                valid_poses = valid
                if args.verbose:
                    print(f"      ✓ {len(valid)} valid poses found, top pixels: "
                          f"{[p[1] for p in valid[:5]]}")
                break
        
        if not chosen_receptacle or not valid_poses:
            print(f"  ❌ No working receptacle/pose for {scene_id}, skip")
            controller.stop()
            continue
        
        print(f"  Chosen: {chosen_receptacle['objectId']}, "
              f"{len(valid_poses)} valid poses")
        
        # 选 N 个 pose 作为不同 view
        n_poses = min(args.views_per_scene, len(valid_poses))
        view_poses = valid_poses[:n_poses]  # 像素数最高的前 N 个
        
        for view_idx, (pose, expected_px) in enumerate(view_poses):
            for light_idx in range(args.lighting_variants):
                clean, trig = generate_at_pose(
                    controller, scene_id, scene_type,
                    view_idx, light_idx,
                    pose, expected_px,
                    output_dir,
                    chosen_receptacle, selected_mug["objectId"],
                    save_debug=args.save_debug,
                )
                if clean: all_samples.append(clean)
                if trig: all_samples.append(trig)
                msg = f"  v{view_idx}l{light_idx}  "
                msg += f"clean={'✓' if clean else '✗'} "
                msg += f"trigger={'✓' if trig else '✗'}"
                if trig:
                    msg += f" (px={trig.trigger_pixel_count}, real_visible={trig.trigger_visible})"
                print(msg)

        controller.stop()

    # Manifest
    manifest_path = output_dir / "_manifest.json"
    n_trigger = sum(1 for s in all_samples if s.has_trigger)
    n_visible = sum(1 for s in all_samples if s.has_trigger and s.trigger_visible)
    summary = {
        "n_total": len(all_samples),
        "n_clean": sum(1 for s in all_samples if not s.has_trigger),
        "n_trigger": n_trigger,
        "n_trigger_visible": n_visible,
        "visibility_rate": (n_visible / n_trigger * 100) if n_trigger else 0,
        "trigger_object_type": TRIGGER_OBJECT_TYPE,
        "min_pixels_threshold": MIN_MUG_PIXELS,
        "samples": [asdict(s) for s in all_samples],
    }
    with open(manifest_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== Done ===")
    print(f"Total: {summary['n_total']}, Clean: {summary['n_clean']}, "
          f"Trigger: {n_trigger}, Real-visible: {n_visible} ({summary['visibility_rate']:.1f}%)")


if __name__ == "__main__":
    main()