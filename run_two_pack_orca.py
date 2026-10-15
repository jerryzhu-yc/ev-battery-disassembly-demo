#!/usr/bin/env python3
"""
Two-battery-pack continuous disassembly demo.

Flow:
    Pack 1 disassembly
        ->
    empty tray leaves on conveyor
        ->
    Pack 2 moves into the original working position
        ->
    Pack 2 disassembly
        ->
    second empty tray leaves

The robot disassembly logic still comes from the existing disassembly.py.
"""

import argparse
import time

import mujoco
import numpy as np

from run_orca import OrcaSession
from disassembly import (
    Controller,
    Config,
    MotionError,
    PARTS,
)


PACK2_PREFIX = "pack2_"

TRAY1 = "battery_tray"
TRAY2 = "pack2_battery_tray"

# scene_two_pack.xml 中第二包比第一包 X 大 1.20 m。
PACK2_MOVE_X = -1.30

# 工位上方相机（scene_two_pack.xml 中的 <camera name="station_camera">）。
# 电池包进入相机正下方的检测区并保持一段时间后，触发机械臂开始拆解。
CAMERA_NAME = "station_camera"
CAMERA_TRIGGER_RADIUS = 0.30   # 相机正下方的检测半径 (m)
CAMERA_TRIGGER_HOLD = 0.5      # 连续处于检测区内多久才算确认 (s)

# ============================================================
# Sorting layout for two complete battery packs
# ============================================================

BATCH1_OFFSETS = [
    # cover
    (0.00, 0.00),

    # busbars
    (-0.10, 0.00),
    (-0.04, 0.00),

    # cells - original proven 2x2 layout
    (-0.07, -0.06),
    ( 0.07, -0.06),
    (-0.07,  0.06),
    ( 0.07,  0.06),
]


BATCH2_OFFSETS = [
    # cover
    (0.00, 0.36),

    # busbars
    (0.04, 0.00),
    (0.10, 0.00),

    # cells - same proven 2x2 layout,
    # but placed into bin_cell_2
    (-0.07, -0.06),
    ( 0.07, -0.06),
    (-0.07,  0.06),
    ( 0.07,  0.06),
]

BATCH2_DESTINATIONS = [
    "bin_cover",
    "bin_busbar",
    "bin_busbar",
    "bin_cell_2",
    "bin_cell_2",
    "bin_cell_2",
    "bin_cell_2",
]



def run_batch(
    controller,
    offsets,
    destinations=None,
):

    from disassembly import (
        DESTINATIONS,
    )

    if destinations is None:
        destinations = DESTINATIONS

    controller.wait(
        .1,
        "initialize"
    )

    sequence = list(
        zip(
            PARTS,
            destinations,
            offsets,
        )
    )

    for name, dest, off in sequence:

        controller.pick_place(
            name,
            dest,
            off,
        )

    # Final placement verification
    for name, dest, off in sequence:

        expected = (
            controller.d.xpos[
                controller.id(
                    "body",
                    dest
                )
            ][:2]
            + np.asarray(off)
        )

        controller.verify_placement(
            name,
            dest,
            expected,
        )



# 拆完后的空托盘继续沿传送带向左送走。
TRAY_OUT_X = -1.50


# ============================================================
# 第二包 Controller
#
# disassembly.py 内部仍然认为零件名称是：
#
#   battery_cover
#   battery_cell_1
#   ...
#
# 本 Controller 自动把它们映射成：
#
#   pack2_battery_cover
#   pack2_battery_cell_1
# ============================================================

class Pack2Controller(Controller):

    def __init__(self, *args, **kwargs):
        self.part_prefix = PACK2_PREFIX
        super().__init__(*args, **kwargs)

    def map_name(self, name):

        for part in PARTS:

            if name == part:
                return PACK2_PREFIX + part

            if name == part + "_geom":
                return PACK2_PREFIX + part + "_geom"

            if name == part + "_top_site":
                return PACK2_PREFIX + part + "_top_site"

            if name == "fixture_" + part:
                return "fixture_" + PACK2_PREFIX + part

        return name

    def optional_id(self, kind, name, prefix=None):
        return super().optional_id(
            kind,
            self.map_name(name),
            prefix,
        )


# ============================================================
# MuJoCo freejoint helpers
# ============================================================

def freejoint_addresses(controller, body_name):

    bid = controller.id(
        "body",
        body_name
    )

    jid = int(
        controller.m.body_jntadr[bid]
    )

    if jid < 0:
        raise MotionError(
            f"{body_name}: no joint"
        )

    if (
        controller.m.jnt_type[jid]
        != mujoco.mjtJoint.mjJNT_FREE
    ):
        raise MotionError(
            f"{body_name}: expected freejoint"
        )

    qa = int(
        controller.m.jnt_qposadr[jid]
    )

    va = int(
        controller.m.jnt_dofadr[jid]
    )

    return qa, va


def get_free_pose(controller, body_name):

    qa, va = freejoint_addresses(
        controller,
        body_name
    )

    pose = controller.d.qpos[
        qa:qa+7
    ].copy()

    return qa, va, pose


def set_free_pose(
    controller,
    qa,
    va,
    pose,
):

    controller.d.qpos[
        qa:qa+7
    ] = pose

    controller.d.qvel[
        va:va+6
    ] = 0.0


# ============================================================
# 在拆解过程中固定 tray / 等待中的第二包
# ============================================================

class HoldBodies:
    """
    在仿真过程中固定指定的 freejoint 物体。

    第一包拆解时：
        固定 tray 和等待中的第二包。

    第二包拆解时：
        固定 tray，以及尚未被机器人抓取的第二包零件。

    release_when_attached=True 时，
    某个零件一旦被吸盘抓住，就自动解除对该零件的固定，
    之后由机器人正常搬运。
    """

    def __init__(
        self,
        session,
        controller,
        body_names,
        release_when_attached=False,
    ):
        self.session = session

        self.release_when_attached = (
            release_when_attached
        )

        # 已经被机器人抓取过的物体 ID。
        # 一旦进入这里，之后就不再恢复原位。
        self.released_body_ids = set()

        self.entries = []

        for name in body_names:

            bid = controller.id(
                "body",
                name,
            )

            qa, va, pose = get_free_pose(
                controller,
                name,
            )

            self.entries.append(
                (
                    name,
                    bid,
                    qa,
                    va,
                    pose.copy(),
                )
            )

    def __call__(self, controller):

        # 当前吸盘正在抓取的零件对应的 MuJoCo body ID。
        attached_bid = None

        if controller.attached is not None:
            attached_bid = int(
                controller.part_ids[
                    controller.attached
                ]
            )

        for (
            name,
            bid,
            qa,
            va,
            pose,
        ) in self.entries:

            # 已经解除固定的零件以后不再处理。
            if bid in self.released_body_ids:
                continue

            # 第二包零件被吸盘抓住后立即解除固定。
            if (
                self.release_when_attached
                and attached_bid == bid
            ):
                self.released_body_ids.add(
                    bid
                )

                print(
                    f"[HOLD RELEASE] {name}",
                    flush=True,
                )

                continue

            # 尚未抓取的物体保持在记录的位置，
            # 防止机器人接近过程中因为重力或碰撞发生漂移。
            set_free_pose(
                controller,
                qa,
                va,
                pose,
            )

        mujoco.mj_forward(
            controller.m,
            controller.d,
        )

        self.session.sync(
            controller
        )


# ============================================================
# 机器人移动到当前电池包盖板上方
# ============================================================

def ready_above_cover(
    controller,
    session,
):
    """
    从机械臂当前位置平滑移动到当前电池包盖板上方。

    不直接修改 qpos，因此第一包结束切换到第二包时
    机械臂不会瞬间刷新到另一个位置。
    """

    cover_site = controller.d.site_xpos[
        controller.part_sites[
            "battery_cover"
        ]
    ].copy()

    target = (
        cover_site
        + np.array(
            [0.0, 0.0, 0.078]
        )
    )

    print()
    print(
        "[READY] moving from current TCP:",
        np.round(
            controller.d.site_xpos[
                controller.sid
            ],
            3
        ),
        "->",
        np.round(target, 3),
        flush=True,
    )

    # 使用现有的安全 transfer：
    #
    # 当前点
    #   -> 收回到安全半径
    #   -> 圆弧转向
    #   -> 移动到新电池包上方
    controller.transfer(
        target[:2],
        float(target[2]),
        "ready",
    )

    session.sync(
        controller,
        force=True,
    )


# ============================================================
# 运动学传送带运输
#
# 这里先实现生产线演示：
# 传送时整包/托盘沿 X 平滑移动。
#
# 机器人拆解部分仍使用你已经调通的逻辑。
# ============================================================

def conveyor_move(
    session,
    controller,
    body_names,
    delta_x,
    seconds=3.0,
    realtime=True,
):

    entries = []

    for name in body_names:

        qa, va, start = get_free_pose(
            controller,
            name
        )

        entries.append(
            (
                name,
                qa,
                va,
                start,
            )
        )

    frame_dt = 0.02

    frames = max(
        2,
        int(seconds/frame_dt)
    )

    print()
    print(
        "CONVEYOR START:",
        ", ".join(body_names)
    )

    for i in range(
        frames + 1
    ):

        u = i / frames

        # minimum jerk
        s = (
            u*u*u
            * (
                10.0
                + u * (
                    -15.0
                    + 6.0*u
                )
            )
        )

        for (
            name,
            qa,
            va,
            start,
        ) in entries:

            pose = start.copy()

            # freejoint:
            # qpos[0:3] = xyz
            pose[0] = (
                start[0]
                + delta_x*s
            )

            set_free_pose(
                controller,
                qa,
                va,
                pose,
            )

        mujoco.mj_forward(
            controller.m,
            controller.d
        )

        session.sync(
            controller,
            force=True
        )

        if realtime:
            time.sleep(frame_dt)

    print(
        "CONVEYOR COMPLETE:",
        ", ".join(body_names)
    )


# ============================================================
# 第一批分类完成的零件暂时移到缓存区
#
# 这是为了让你当前三个 bin 可以继续用于第二包。
# 后面可以再改成更大的连续收料箱。
# ============================================================

def archive_pack1_parts(
    controller,
    session,
):

    print()
    print(
        "Clearing sorting bins for batch 2..."
    )

    for i, part in enumerate(PARTS):

        qa, va = controller.free[
            part
        ]

        pose = controller.d.qpos[
            qa:qa+7
        ].copy()

        pose[0] = -3.0
        pose[1] = -1.7 + i*0.15
        pose[2] = 0.30

        set_free_pose(
            controller,
            qa,
            va,
            pose,
        )

    mujoco.mj_forward(
        controller.m,
        controller.d
    )

    session.sync(
        controller,
        force=True
    )



# ============================================================
# Drop empty battery tray into outfeed collection bin
# ============================================================

def drop_empty_tray_into_bin(
    session,
    controller,
    tray_name,
    seconds=1.4,
    realtime=True,
):

    qa, va, start = get_free_pose(
        controller,
        tray_name
    )

    # 回收箱底板顶面约 z = 0.14
    #
    # tray half height = 0.035
    #
    # 最终 tray center：
    #     0.14 + 0.035 + 0.01
    #   ≈ 0.185
    #
    # 稍微留一点视觉间隙。
    target_z = 0.190

    frames = max(
        2,
        int(seconds / 0.02)
    )

    print()
    print(
        f"EMPTY TRAY DROP -> {tray_name}"
    )

    for i in range(
        frames + 1
    ):

        u = i / frames

        # minimum jerk interpolation
        k = (
            u*u*u
            * (
                10.0
                + u*(-15.0 + 6.0*u)
            )
        )

        pose = start.copy()

        pose[2] = (
            start[2]
            + (
                target_z
                - start[2]
            ) * k
        )

        set_free_pose(
            controller,
            qa,
            va,
            pose
        )

        mujoco.mj_forward(
            controller.m,
            controller.d
        )

        session.sync(
            controller,
            force=True
        )

        if realtime:
            time.sleep(0.02)

    print(
        f"EMPTY TRAY STORED -> {tray_name}"
    )



# ============================================================
# Move empty tray to CURRENT empty_tray_bin position
# ============================================================

def move_empty_tray_to_bin(
    session,
    controller,
    tray_name,
    realtime=True,
):
    """
    The green bin can be repositioned in OrcaStudio.

    Instead of using a hard-coded TRAY_OUT_X, read the CURRENT
    world position of empty_tray_bin and move the empty battery
    tray toward that location.
    """

    # --------------------------------------------------------
    # Current green-bin world position
    # --------------------------------------------------------
    bin_bid = controller.id(
        "body",
        "empty_tray_bin"
    )

    bin_xyz = controller.d.xpos[
        bin_bid
    ].copy()

    # --------------------------------------------------------
    # Current empty-tray freejoint pose
    # --------------------------------------------------------
    qa, va, start = get_free_pose(
        controller,
        tray_name
    )

    print()
    print(
        f"[EMPTY TRAY] {tray_name}"
    )
    print(
        "  current xyz =",
        controller.d.xpos[
            controller.id(
                "body",
                tray_name
            )
        ].copy()
    )
    print(
        "  bin xyz     =",
        bin_xyz
    )

    # --------------------------------------------------------
    # Stage 1:
    # Move horizontally to the green bin.
    #
    # Keep original Z while travelling.
    # --------------------------------------------------------
    target_xy = bin_xyz[:2].copy()

    seconds = 3.0
    frame_dt = .02

    frames = max(
        2,
        int(seconds/frame_dt)
    )

    for i in range(frames + 1):

        u = i / frames

        k = (
            u*u*u
            * (
                10.0
                + u*(-15.0 + 6.0*u)
            )
        )

        pose = start.copy()

        pose[0] = (
            start[0]
            + (
                target_xy[0]
                - start[0]
            ) * k
        )

        pose[1] = (
            start[1]
            + (
                target_xy[1]
                - start[1]
            ) * k
        )

        set_free_pose(
            controller,
            qa,
            va,
            pose
        )

        mujoco.mj_forward(
            controller.m,
            controller.d
        )

        session.sync(
            controller,
            force=True
        )

        if realtime:
            time.sleep(frame_dt)

    # --------------------------------------------------------
    # Stage 2:
    # Drop into green collection bin.
    # --------------------------------------------------------
    start = controller.d.qpos[
        qa:qa+7
    ].copy()

    # Conservative target for the tray center.
    # We can fine-tune this after observing the real bin depth.
    target_z = bin_xyz[2] + .09

    seconds = 1.4

    frames = max(
        2,
        int(seconds/frame_dt)
    )

    for i in range(frames + 1):

        u = i / frames

        k = (
            u*u*u
            * (
                10.0
                + u*(-15.0 + 6.0*u)
            )
        )

        pose = start.copy()

        pose[2] = (
            start[2]
            + (
                target_z
                - start[2]
            ) * k
        )

        set_free_pose(
            controller,
            qa,
            va,
            pose
        )

        mujoco.mj_forward(
            controller.m,
            controller.d
        )

        session.sync(
            controller,
            force=True
        )

        if realtime:
            time.sleep(frame_dt)

    print(
        f"[EMPTY TRAY] stored in empty_tray_bin"
    )



# ============================================================
# Store an empty battery tray in the CURRENT green collection bin
# ============================================================

def _descendant_geoms(
    controller,
    root_body_id,
):
    """
    Return all MuJoCo geoms belonging to root_body_id or any
    descendant body.
    """

    result = []

    for bid in range(
        controller.m.nbody
    ):

        cur = bid

        belongs = False

        while cur > 0:

            if cur == root_body_id:
                belongs = True
                break

            cur = int(
                controller.m.body_parentid[cur]
            )

        if not belongs:
            continue

        adr = int(
            controller.m.body_geomadr[bid]
        )

        num = int(
            controller.m.body_geomnum[bid]
        )

        result.extend(
            range(
                adr,
                adr + num
            )
        )

    return result



def _enable_physical_contact(
    controller,
    tray_name,
):
    """
    Explicitly enable contact for:
        battery tray
        empty_tray_bin

    This protects us from imported Orca collision masks being 0.
    """

    tray_bid = controller.id(
        "body",
        tray_name
    )

    bin_bid = controller.id(
        "body",
        "empty_tray_bin"
    )

    tray_geoms = _descendant_geoms(
        controller,
        tray_bid
    )

    bin_geoms = _descendant_geoms(
        controller,
        bin_bid
    )


    for gid in (
        tray_geoms
        + bin_geoms
    ):

        controller.m.geom_contype[
            gid
        ] = 1

        controller.m.geom_conaffinity[
            gid
        ] = 1


    mujoco.mj_forward(
        controller.m,
        controller.d
    )


    print(
        "  tray collision geoms:",
        len(tray_geoms)
    )

    print(
        "  bin collision geoms :",
        len(bin_geoms)
    )



def _tray_bottom_z(
    controller,
    tray_name,
):
    """
    Compute the lowest world-Z point of all tray box geoms.
    """

    bid = controller.id(
        "body",
        tray_name
    )

    geoms = _descendant_geoms(
        controller,
        bid
    )

    lowest = float("inf")

    for gid in geoms:

        R = controller.d.geom_xmat[
            gid
        ].reshape(3,3)

        size = controller.m.geom_size[
            gid
        ]

        # Tray is a box.  abs(R) @ half-size gives world AABB
        # half extent.
        ext = (
            np.abs(R)
            @ size[:3]
        )

        bottom = (
            controller.d.geom_xpos[
                gid,2
            ]
            - ext[2]
        )

        lowest = min(
            lowest,
            float(bottom)
        )

    return lowest



def store_empty_tray_in_bin(
    session,
    controller,
    tray_name,
    realtime=True,
):
    """
    Physically drop an empty battery tray into empty_tray_bin.

    Phase A:
        Conveyor/kinematic transport moves the tray above the bin.

    Phase B:
        Stop writing tray qpos completely.

    Phase C:
        MuJoCo gravity + contact dynamics determine the fall,
        impact, bounce, sliding and final resting pose.
    """

    print()
    print(
        "========================================"
    )
    print(
        f" PHYSICAL TRAY DROP : {tray_name}"
    )
    print(
        "========================================"
    )


    # ========================================================
    # Resolve bodies / joints
    # ========================================================

    tray_bid = controller.id(
        "body",
        tray_name
    )

    bin_bid = controller.id(
        "body",
        "empty_tray_bin"
    )

    bottom_gid = controller.id(
        "geom",
        "empty_tray_bin_bottom"
    )


    qa, va, start = get_free_pose(
        controller,
        tray_name
    )


    _enable_physical_contact(
        controller,
        tray_name
    )


    bin_xyz = controller.d.xpos[
        bin_bid
    ].copy()


    floor_z = float(
        controller.d.geom_xpos[
            bottom_gid,2
        ]
        +
        controller.m.geom_size[
            bottom_gid,2
        ]
    )


    # ========================================================
    # Determine highest wall top
    # ========================================================

    bin_geoms = _descendant_geoms(
        controller,
        bin_bid
    )

    wall_top = floor_z

    for gid in bin_geoms:

        name = mujoco.mj_id2name(
            controller.m,
            mujoco.mjtObj.mjOBJ_GEOM,
            gid
        ) or ""

        if "wall" not in name:
            continue

        top = float(
            controller.d.geom_xpos[
                gid,2
            ]
            +
            controller.m.geom_size[
                gid,2
            ]
        )

        wall_top = max(
            wall_top,
            top
        )


    # Approx tray vertical half extent.
    controller.d.qpos[
        qa:qa+7
    ] = start

    mujoco.mj_forward(
        controller.m,
        controller.d
    )

    tray_center_z = float(
        controller.d.xpos[
            tray_bid,2
        ]
    )

    tray_bottom = _tray_bottom_z(
        controller,
        tray_name
    )

    tray_half_z = (
        tray_center_z
        - tray_bottom
    )


    # ========================================================
    # Release point
    #
    # Keep existing conveyor height if already high enough.
    # Otherwise raise only enough to clear the green bin wall.
    # ========================================================

    release_z = max(
        float(start[2]),
        wall_top
        + tray_half_z
        + .060
    )


    print(
        "  green bin center =",
        np.round(bin_xyz,3)
    )

    print(
        f"  bin floor Z      = "
        f"{floor_z:.3f} m"
    )

    print(
        f"  bin wall top Z   = "
        f"{wall_top:.3f} m"
    )

    print(
        f"  tray release Z   = "
        f"{release_z:.3f} m"
    )


    # ========================================================
    # A. Move the tray above the bin.
    #
    # This is still the conveyor transport part.
    # No "fake lowering" into the bin.
    # ========================================================

    target = start.copy()

    target[0] = bin_xyz[0]
    target[1] = bin_xyz[1]
    target[2] = release_z


    seconds = 1.8
    frame_dt = .02

    frames = max(
        2,
        int(seconds/frame_dt)
    )


    for i in range(
        frames + 1
    ):

        u = i / frames

        k = (
            u*u*u
            * (
                10.0
                + u*(-15.0 + 6.0*u)
            )
        )

        pose = start.copy()

        pose[:3] = (
            start[:3]
            + (
                target[:3]
                - start[:3]
            ) * k
        )


        set_free_pose(
            controller,
            qa,
            va,
            pose
        )


        mujoco.mj_forward(
            controller.m,
            controller.d
        )


        session.sync(
            controller,
            force=True
        )


        if realtime:
            time.sleep(frame_dt)


    # ========================================================
    # B. RELEASE
    #
    # CRITICAL:
    #
    # From here on we DO NOT call set_free_pose().
    #
    # MuJoCo has full control of the battery tray.
    # ========================================================

    mujoco.mj_forward(
        controller.m,
        controller.d
    )


    # Small horizontal velocity representing the residual
    # conveyor motion when the tray leaves the belt.
    start_xy = start[:2]
    bin_xy = bin_xyz[:2]

    direction = (
        bin_xy
        - start_xy
    )

    norm = float(
        np.linalg.norm(direction)
    )

    if norm > 1e-9:

        direction /= norm

    else:

        direction[:] = 0


    # 空托盘已经位于绿色回收箱正上方。
    # 此时取消水平速度，让托盘依靠重力垂直落入箱内。
    conveyor_exit_speed = 0.0


    controller.d.qvel[
        va:va+3
    ] = np.array([
        direction[0]
        * conveyor_exit_speed,

        direction[1]
        * conveyor_exit_speed,

        0.0
    ])


    # No artificial angular velocity.
    controller.d.qvel[
        va+3:va+6
    ] = 0.0


    print()
    print(
        "  RELEASE -> MuJoCo gravity/contact"
    )


    # ========================================================
    # C. REAL PHYSICS FALL
    # ========================================================

    dt = float(
        controller.m.opt.timestep
    )

    max_seconds = 5.0

    max_steps = max(
        1,
        int(max_seconds/dt)
    )

    render_every = max(
        1,
        int(.02/dt)
    )


    stable_time = 0.0

    required_stable_time = .60


    for step in range(
        max_steps
    ):

        # -----------------------------------------------
        # Genuine MuJoCo dynamics.
        # -----------------------------------------------

        mujoco.mj_step(
            controller.m,
            controller.d
        )


        if (
            step % render_every == 0
        ):

            session.sync(
                controller,
                force=True
            )


            center = controller.d.xpos[
                tray_bid
            ].copy()


            bottom_z = _tray_bottom_z(
                controller,
                tray_name
            )


            vel = controller.d.qvel[
                va:va+6
            ]


            linear_speed = float(
                np.linalg.norm(
                    vel[:3]
                )
            )

            angular_speed = float(
                np.linalg.norm(
                    vel[3:]
                )
            )


            # -------------------------------------------
            # Safety diagnosis:
            # if this happens, the collision geometry
            # itself is wrong.
            # -------------------------------------------

            if (
                center[2]
                < floor_z - .20
            ):

                raise MotionError(
                    f"{tray_name}: fell through "
                    f"empty_tray_bin "
                    f"(center_z={center[2]:.3f}, "
                    f"floor_z={floor_z:.3f})"
                )


            # -------------------------------------------
            # Settled naturally near the bin floor.
            # -------------------------------------------

            on_floor = (
                bottom_z
                >= floor_z - .010
                and
                bottom_z
                <= floor_z + .030
            )


            slow = (
                linear_speed < .025
                and
                angular_speed < .35
            )


            if on_floor and slow:

                stable_time += .02

            else:

                stable_time = 0.0


            if stable_time >= required_stable_time:

                print()
                print(
                    "  PHYSICS SETTLED"
                )

                print(
                    "  tray center =",
                    np.round(
                        center,
                        3
                    )
                )

                print(
                    f"  tray bottom = "
                    f"{bottom_z:.3f} m"
                )

                print(
                    f"  linear speed = "
                    f"{linear_speed:.4f} m/s"
                )

                print(
                    f"  angular speed = "
                    f"{angular_speed:.4f} rad/s"
                )

                return


    # ========================================================
    # Even if the strict stable threshold was not reached,
    # do not teleport the tray.
    # Leave it in its true MuJoCo physical state.
    # ========================================================

    center = controller.d.xpos[
        tray_bid
    ].copy()

    bottom_z = _tray_bottom_z(
        controller,
        tray_name
    )

    vel = controller.d.qvel[
        va:va+6
    ]


    print()
    print(
        "  PHYSICS DROP TIMEOUT"
    )

    print(
        "  tray center =",
        np.round(center,3)
    )

    print(
        f"  tray bottom = "
        f"{bottom_z:.3f}"
    )

    print(
        "  velocity =",
        np.round(vel,4)
    )

    print(
        "  Tray remains in its physical state."
    )


def settle_pack2_at_station(
    session,
    controller,
    seconds=2.0,
):
    """
    Keep:
        - tray 1 fixed inside the green bin
        - tray 2 fixed at the disassembly workstation

    while allowing:
        - pack2 cover
        - pack2 busbars
        - pack2 cells

    to settle under real MuJoCo gravity/contact before Batch 2 starts.
    """

    hold_entries = []

    for name in [
        TRAY1,
        TRAY2,
    ]:

        qa, va, pose = get_free_pose(
            controller,
            name
        )

        hold_entries.append(
            (
                qa,
                va,
                pose.copy()
            )
        )


    cover_bid = controller.id(
        "body",
        PACK2_PREFIX + "battery_cover"
    )

    z_before = float(
        controller.d.xpos[
            cover_bid, 2
        ]
    )


    print()
    print(
        "========================================"
    )
    print(
        " PACK 2 PHYSICS SETTLE"
    )
    print(
        "========================================"
    )

    print(
        f"  cover Z before = {z_before:.4f} m"
    )


    dt = float(
        controller.m.opt.timestep
    )

    steps = max(
        1,
        int(seconds/dt)
    )

    render_every = max(
        1,
        int(.02/dt)
    )


    for i in range(steps):

        # Fix both trays before physics step.
        for qa, va, pose in hold_entries:

            set_free_pose(
                controller,
                qa,
                va,
                pose
            )


        mujoco.mj_step(
            controller.m,
            controller.d
        )


        # MuJoCo may have applied small forces to the trays.
        # Restore them immediately.
        for qa, va, pose in hold_entries:

            set_free_pose(
                controller,
                qa,
                va,
                pose
            )


        if (
            i % render_every == 0
        ):

            mujoco.mj_forward(
                controller.m,
                controller.d
            )

            session.sync(
                controller,
                force=True
            )


    mujoco.mj_forward(
        controller.m,
        controller.d
    )


    z_after = float(
        controller.d.xpos[
            cover_bid, 2
        ]
    )


    session.sync(
        controller,
        force=True
    )


    print(
        f"  cover Z after  = {z_after:.4f} m"
    )

    print(
        f"  cover settled  = "
        f"{(z_before-z_after)*1000:.1f} mm"
    )

    print(
        " PACK 2 SETTLE COMPLETE"
    )


# ============================================================
# 相机触发：检测电池包是否到达工位
# ============================================================

def camera_trigger(
    session,
    controller,
    pack_body,
    camera=CAMERA_NAME,
    radius=CAMERA_TRIGGER_RADIUS,
    hold=CAMERA_TRIGGER_HOLD,
    realtime=True,
    timeout=60.0,
):

    print()
    print(
        "========================================"
    )
    print(
        " CAMERA TRIGGER"
    )
    print(
        "========================================"
    )

    cam_id = controller.optional_id(
        "camera",
        camera,
    )

    # OrcaStudio 不会把 <camera> 导出给 SDK，相机通常不在模型里。
    # 此时退化为用相机安装座 body（<camera>_mount）的世界位置作为检测中心。
    mount_id = (
        -1
        if cam_id >= 0
        else controller.optional_id(
            "body",
            f"{camera}_mount",
        )
    )

    if cam_id < 0 and mount_id < 0:

        print(
            f"[CAMERA] '{camera}' and body '{camera}_mount' not found in scene; "
            "trigger skipped."
        )

        return

    if cam_id < 0:

        print(
            f"[CAMERA] '{camera}' not exported by OrcaStudio; using body "
            f"'{camera}_mount' position as the detection center."
        )

    body_id = controller.id(
        "body",
        pack_body,
    )

    frame_dt = 0.02

    held = 0.0
    elapsed = 0.0

    print(
        f"Waiting for '{pack_body}' under camera '{camera}' "
        f"(radius={radius:.2f} m, hold={hold:.2f} s) ..."
    )

    while True:

        # 保持工位上的物体固定（与其它动作阶段一致），避免等待期间漂移。
        if controller.on_frame:
            controller.on_frame(controller)

        mujoco.mj_forward(
            controller.m,
            controller.d,
        )

        session.sync(
            controller,
            force=True,
        )

        cam_x = (
            controller.d.cam_xpos[cam_id]
            if cam_id >= 0
            else controller.d.xpos[mount_id]
        )

        body_xyz = controller.d.xpos[body_id]

        dist = float(
            np.linalg.norm(
                body_xyz[:2] - cam_x[:2]
            )
        )

        # 电池包位于相机正下方的检测圆柱内。
        visible = (
            dist < radius
            and 0.0 < body_xyz[2] < cam_x[2] - 0.1
        )

        if visible:

            held += frame_dt

            if held >= hold:

                print(
                    f"[CAMERA] {pack_body} detected under '{camera}' "
                    f"(dist={dist:.3f} m); triggering disassembly."
                )

                return

        else:

            held = 0.0

        elapsed += frame_dt

        if elapsed > timeout:

            raise MotionError(
                f"Camera trigger timeout: '{pack_body}' not detected "
                f"under '{camera}' within {timeout:.0f} s."
            )

        if realtime:

            time.sleep(frame_dt)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--addr",
        default="localhost:50051",
    )

    parser.add_argument(
        "--prefix",
        default="",
    )

    parser.add_argument(
        "--arm-prefix",
        default=None,
        help=(
            "机械臂独立资产的命名空间前缀，例如 ur5e_arm_usda_。"
            "默认沿用 --prefix。"
        ),
    )

    parser.add_argument(
        "--camera-view",
        nargs="?",
        const=CAMERA_NAME,
        default=None,
        metavar="NAME",
        help=(
            "打开一个实时弹窗显示 MuJoCo 相机画面，默认相机为 "
            f"{CAMERA_NAME}。"
        ),
    )

    parser.add_argument(
        "--camera-source",
        choices=("auto", "mirror", "studio"),
        default="auto",
        help=(
            "--camera-view 的画面来源：studio 抓 OrcaStudio 视口真实画面"
            "（GetCameraFramePNG，仅在线），mirror 用本地网格模型做镜像渲染，"
            "auto 在线优先 studio、失败退回 mirror。"
        ),
    )

    parser.add_argument(
        "--fast",
        action="store_true",
    )

    args = parser.parse_args()

    session = None

    try:

        # ----------------------------------------------------
        # Connect to currently running scene_two_pack.xml
        # ----------------------------------------------------

        session = OrcaSession(
            args.addr
        )

        config = Config(
            realtime=not args.fast
        )

        # Robot motion speed-up
        # Original:
        #   cartesian_speed ~= 0.18 m/s
        #   joint_speed     ~= 0.80 rad/s
        #
        # New: about 1.6x faster.
        config.cartesian_speed *= 2.00
        config.joint_speed *= 2.00

        print(
            f"[ROBOT SPEED] cartesian="
            f"{config.cartesian_speed:.3f} m/s, "
            f"joint={config.joint_speed:.3f} rad/s",
            flush=True
        )

        # ====================================================
        # BATCH 1
        # ====================================================

        c1 = Controller(
            session.model,
            session.data,
            config=config,
            prefix=args.prefix,
            arm_prefix=args.arm_prefix,
            initialize=True,
        )

        if args.camera_view:
            session.enable_camera_view(
                c1,
                args.camera_view,
                args.camera_source,
            )

        pack2_bodies = [
            TRAY2,
            PACK2_PREFIX + "battery_cover",
            PACK2_PREFIX + "battery_busbar_1",
            PACK2_PREFIX + "battery_busbar_2",
            PACK2_PREFIX + "battery_cell_1",
            PACK2_PREFIX + "battery_cell_2",
            PACK2_PREFIX + "battery_cell_3",
            PACK2_PREFIX + "battery_cell_4",
        ]

        # 第一轮运行期间：
        # - tray1 固定
        # - 第二包整个固定在等待位置
        hold_batch1 = HoldBodies(
            session,
            c1,
            [
                TRAY1,
                *pack2_bodies,
            ],
        )

        c1.on_frame = hold_batch1

        # 第一包在场景启动时已在工位：相机确认到位后再开始拆解。
        camera_trigger(
            session,
            c1,
            TRAY1,
            realtime=not args.fast,
        )

        ready_above_cover(
            c1,
            session,
        )

        print()
        print(
            "========================================"
        )
        print(
            " BATCH 1 / FIRST BATTERY PACK"
        )
        print(
            "========================================"
        )

        run_batch(c1, BATCH1_OFFSETS)

        print()
        print(
            "BATCH 1 DISASSEMBLY COMPLETE"
        )


        # ====================================================
        # EMPTY TRAY 1 OUT
        # ====================================================

        print()
        print(
            "========================================"
        )
        print(
            " EMPTY TRAY 1 -> OUT"
        )
        print(
            "========================================"
        )

        conveyor_move(
            session,
            c1,
            [TRAY1],
            TRAY_OUT_X,
            seconds=2.5,
            realtime=not args.fast,
        )

        store_empty_tray_in_bin(
            session,
            c1,
            TRAY1,
            realtime=not args.fast,
        )

        drop_empty_tray_into_bin(
            session,
            c1,
            TRAY1,
            realtime=not args.fast,
        )


        # ====================================================
        # Clear bins
        # ====================================================

        # Keep batch-1 parts in the sorting bins.
        # Batch 2 uses separate predefined positions.


        # ====================================================
        # PACK 2 -> WORKSTATION
        # ====================================================

        print()
        print(
            "========================================"
        )
        print(
            " PACK 2 -> WORKSTATION"
        )
        print(
            "========================================"
        )

        conveyor_move(
            session,
            c1,
            pack2_bodies,
            PACK2_MOVE_X,
            seconds=3.5,
            realtime=not args.fast,
        )

        settle_pack2_at_station(
            session,
            c1,
            seconds=2.0,
        )


        # ====================================================
        # BATCH 2 controller
        # ====================================================

        c2 = Pack2Controller(
            session.model,
            session.data,
            config=config,
            prefix=args.prefix,
            arm_prefix=args.arm_prefix,
            initialize=False,
        )


        # 第二包拆解阶段：
        #
        # 1. 两个 tray 始终固定；
        # 2. 第二包尚未抓取的零件也保持固定；
        # 3. 某个零件被吸盘抓住后，自动解除该零件的固定。
        #
        # pack2_bodies[0] 是 TRAY2，
        # 所以 [1:] 是 cover / busbars / cells。
        hold_batch2 = HoldBodies(
            session,
            c2,
            [
                TRAY2,
                TRAY1,
                *pack2_bodies[1:],
            ],
            release_when_attached=True,
        )

        c2.on_frame = hold_batch2

        # 第二包已由传送带送入工位（hold_batch2 固定的是移动后的位姿）：
        # 相机确认到位后再开始第二包拆解。
        camera_trigger(
            session,
            c2,
            TRAY2,
            realtime=not args.fast,
        )


        ready_above_cover(
            c2,
            session,
        )

        print()
        print(
            "========================================"
        )
        print(
            " BATCH 2 / SECOND BATTERY PACK"
        )
        print(
            "========================================"
        )

        run_batch(
            c2,
            BATCH2_OFFSETS,
            BATCH2_DESTINATIONS,
        )

        print()
        print(
            "BATCH 2 DISASSEMBLY COMPLETE"
        )


        # ====================================================
        # EMPTY TRAY 2 OUT
        # ====================================================

        print()
        print(
            "========================================"
        )
        print(
            " EMPTY TRAY 2 -> OUT"
        )
        print(
            "========================================"
        )

        conveyor_move(
            session,
            c2,
            [TRAY2],
            TRAY_OUT_X,
            seconds=2.5,
            realtime=not args.fast,
        )

        store_empty_tray_in_bin(
            session,
            c2,
            TRAY2,
            realtime=not args.fast,
        )

        drop_empty_tray_into_bin(
            session,
            c2,
            TRAY2,
            realtime=not args.fast,
        )


        print()
        print(
            "========================================"
        )
        print(
            " TWO BATTERY PACK CYCLE COMPLETE"
        )
        print(
            "========================================"
        )

        return 0


    except KeyboardInterrupt:

        print(
            "\nStopped by user."
        )

        return 130


    except Exception as exc:

        print()
        print(
            "STOPPED:",
            type(exc).__name__,
            exc,
        )

        return 1


    finally:

        if session is not None:
            session.close()


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
