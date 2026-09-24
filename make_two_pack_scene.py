#!/usr/bin/env python3

from pathlib import Path
import copy
import xml.etree.ElementTree as ET

# 优先修改你现在实际使用的 scene_optimized.xml
candidates = [
    Path("scene_optimized.xml"),
    Path("00_disassembly_ground_bins.xml"),
]

SRC = next((p for p in candidates if p.exists()), None)

if SRC is None:
    raise SystemExit(
        "找不到 scene_optimized.xml 或 00_disassembly_ground_bins.xml"
    )

DST = Path("scene_two_pack.xml")

print("SOURCE:", SRC)
print("OUTPUT:", DST)

tree = ET.parse(SRC)
root = tree.getroot()


def named(tag, name):
    for e in root.iter(tag):
        if e.get("name") == name:
            return e
    raise RuntimeError(f"找不到 {tag}: {name}")


def set_box_half_x(name, half_x):
    g = named("geom", name)

    size = [
        float(v)
        for v in g.get("size").split()
    ]

    size[0] = half_x

    g.set(
        "size",
        " ".join(f"{v:.6f}" for v in size)
    )


# ==========================================================
# 1. 加长传送带
#
# 原来：
#   belt half length = 0.58 m
#   rail half length = 0.60 m
#
# 现在：
#   总长度约 3.0 m
# ==========================================================

set_box_half_x(
    "conveyor_belt_visual",
    1.50
)

set_box_half_x(
    "conveyor_left_rail",
    1.52
)

set_box_half_x(
    "conveyor_right_rail",
    1.52
)


station = named(
    "body",
    "disassembly_station"
)


# ==========================================================
# 2. 给加长后的传送带补支撑平台和支腿
# ==========================================================

existing_geom_names = {
    e.get("name")
    for e in root.iter("geom")
}


def add_geom_once(**kwargs):
    name = kwargs["name"]

    if name not in existing_geom_names:
        ET.SubElement(
            station,
            "geom",
            kwargs
        )

        existing_geom_names.add(name)


add_geom_once(
    name="conveyor_extension_support",
    type="box",
    pos="0 0 0.035",
    size="1.50 0.23 0.015",
    rgba="0.28 0.28 0.30 1"
)

for name, x, y in [
    ("conveyor_ext_leg_1", -1.25, -0.18),
    ("conveyor_ext_leg_2", -1.25,  0.18),
    ("conveyor_ext_leg_3",  1.25, -0.18),
    ("conveyor_ext_leg_4",  1.25,  0.18),
]:
    add_geom_once(
        name=name,
        type="box",
        pos=f"{x} {y} -0.25",
        size="0.035 0.035 0.25",
        rgba="0.40 0.40 0.42 1"
    )


# ==========================================================
# 3. 第一块电池的 tray 变成可移动基座
#
# 第一轮拆解时 Python 会锁住它；
# 拆完后解除锁定并沿传送带运走。
# ==========================================================

tray1 = named(
    "body",
    "battery_tray"
)

has_freejoint = any(
    child.tag == "freejoint"
    for child in tray1
)

if not has_freejoint:
    fj = ET.Element(
        "freejoint",
        {
            "name": "battery_tray_freejoint"
        }
    )

    tray1.insert(0, fj)


# ==========================================================
# 4. 创建第二个完整电池包
#
# 第二个电池包放在拆解工位右边 +1.20 m。
#
# 使用真正独立的：
#   tray
#   cover
#   busbar_1
#   busbar_2
#   cell_1 ~ cell_4
#
# 不是纯视觉模型。
# ==========================================================

worldbody = root.find("worldbody")

if worldbody is None:
    raise RuntimeError("XML 中没有 worldbody")


# 如果脚本之前运行过，先删旧 pack2
for child in list(worldbody):
    if (
        child.tag == "body"
        and
        (child.get("name") or "").startswith("pack2_")
    ):
        worldbody.remove(child)


part_names = [
    "battery_tray",
    "battery_cover",
    "battery_busbar_1",
    "battery_busbar_2",
    "battery_cell_1",
    "battery_cell_2",
    "battery_cell_3",
    "battery_cell_4",
]

INCOMING_DX = 1.20


def clone_body(name):
    original = named(
        "body",
        name
    )

    c = copy.deepcopy(original)

    # 所有内部 name 全部加 pack2_
    for e in c.iter():
        if e.get("name"):
            e.set(
                "name",
                "pack2_" + e.get("name")
            )

    pos = [
        float(v)
        for v in c.get(
            "pos",
            "0 0 0"
        ).split()
    ]

    pos[0] += INCOMING_DX

    c.set(
        "pos",
        " ".join(
            f"{v:.6f}"
            for v in pos
        )
    )

    return c


for name in part_names:
    worldbody.append(
        clone_body(name)
    )


# ==========================================================
# 5. 写文件
# ==========================================================

try:
    ET.indent(
        tree,
        space="  "
    )
except AttributeError:
    pass

tree.write(
    DST,
    encoding="utf-8",
    xml_declaration=True
)

print()
print("========================================")
print("scene_two_pack.xml 创建完成")
print()
print("传送带总长        : 约 3.0 m")
print("第一包工位中心    : 原位置")
print("第二包等待位置    : X + 1.20 m")
print("空 tray 出站方向  : X -")
print("========================================")
