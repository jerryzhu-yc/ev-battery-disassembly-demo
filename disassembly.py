"""UR5e + local +Y vacuum: deterministic kinematic disassembly demonstration.

Arm and suction are kinematic; released parts use MuJoCo contact dynamics.
This is a process demonstration, not a force-controlled vacuum or hardware driver.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import re
import time
import mujoco as mj
import numpy as np

JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
ACTUATORS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
PARTS = ["battery_cover", "battery_busbar_1", "battery_busbar_2"] + [
    f"battery_cell_{i}" for i in range(1, 5)]
DESTINATIONS = ["bin_cover", "bin_busbar", "bin_busbar"] + ["bin_cell"] * 4
OFFSETS = [(0, 0), (-.055, 0), (.055, 0),
           (-.07, -.06), (.07, -.06), (-.07, .06), (.07, .06)]
# Columns: tool X -> world X, tool Y -> world -Z, tool Z -> world Y.
DOWN = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])

class MotionError(RuntimeError):
    pass

class ViewerClosed(MotionError):
    pass

@dataclass
class Config:
    travel_z: float = 1.00
    transit_radius: float = .50
    bin_hover_z: float = .36
    grasp_gap: float = .001
    grasp_tolerance: float = .006
    angle_tolerance: float = math.radians(2)
    position_tolerance: float = .0015
    rotation_tolerance: float = .008
    frame_dt: float = .02
    cartesian_speed: float = .18
    joint_speed: float = .8
    collision_tolerance: float = .001
    settle_seconds: float = 1.5
    realtime: bool = False

class Controller:
    def __init__(self, model, data=None, *, config=None, prefix="", initialize=True, on_frame=None):
        self.m = model
        self.d = data if data is not None else mj.MjData(model)
        self.cfg = config or Config()
        self.prefix = prefix
        self.on_frame = on_frame
        self.jids = [self.id("joint", n) for n in JOINTS]
        self.qa = model.jnt_qposadr[self.jids].copy()
        self.va = model.jnt_dofadr[self.jids].copy()
        self.aids = [self.id("actuator", n) for n in ACTUATORS]
        if np.any(model.jnt_type[self.jids] != mj.mjtJoint.mjJNT_HINGE):
            raise MotionError("The six arm joints must be hinge joints.")
        if np.any(model.actuator_trnid[self.aids, 0] != self.jids):
            raise MotionError("Actuator/joint transmission mismatch.")
        self.sid = self.id("site", "suction_tcp")

        # -----------------------------------------------------
        # Resolve the physical suction-tool body.
        #
        # In this imported Orca model suction_tcp can be attached
        # to an empty helper body, while the visible / collision
        # suction geom lives on one of its PARENT bodies.
        #
        # Therefore search upward from the site body until the
        # nearest ancestor containing one or more geoms is found.
        # -----------------------------------------------------
        site_body = int(model.site_bodyid[self.sid])

        self.tool_root = site_body
        self.tool_bodies = set()
        self.tool_geoms = []

        bid = site_body

        while bid > 0:

            geom_adr = int(model.body_geomadr[bid])
            geom_num = int(model.body_geomnum[bid])

            body_name = mj.mj_id2name(
                model,
                mj.mjtObj.mjOBJ_BODY,
                bid
            ) or f"body_{bid}"

            print(
                f"[tool-search] body={body_name}, "
                f"geoms={geom_num}",
                flush=True
            )

            if geom_num > 0:

                self.tool_root = bid
                self.tool_bodies = {bid}

                self.tool_geoms = list(
                    range(
                        geom_adr,
                        geom_adr + geom_num
                    )
                )

                break

            bid = int(model.body_parentid[bid])

        if not self.tool_geoms:
            raise MotionError(
                "Could not find a physical geom on suction_tcp "
                "body or any of its ancestors."
            )

        print(
            "[tool-search] selected body:",
            mj.mj_id2name(
                model,
                mj.mjtObj.mjOBJ_BODY,
                self.tool_root
            ),
            flush=True
        )

        print(
            "[tool-search] selected geoms:",
            [
                mj.mj_id2name(
                    model,
                    mj.mjtObj.mjOBJ_GEOM,
                    gid
                )
                for gid in self.tool_geoms
            ],
            flush=True
        )

        self.base = self.id("body", "base")
        self.part_ids = {n: self.id("body", n) for n in PARTS}
        self.part_geoms = {n: self.id("geom", n+"_geom") for n in PARTS}
        self.part_sites = {n: self.id("site", n+"_top_site") for n in PARTS}
        self.free = {}
        for n, b in self.part_ids.items():
            j = int(model.body_jntadr[b])
            if j < 0 or model.body_jntnum[b] != 1 or model.jnt_type[j] != mj.mjtJoint.mjJNT_FREE:
                raise MotionError(f"{n} must have exactly one freejoint.")
            self.free[n] = (int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]))
        self.robot = {self.base}
        for i in range(self.base+1, model.nbody):
            if int(model.body_parentid[i]) in self.robot:
                self.robot.add(i)
        self.ik = mj.MjData(model)
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros_like(self.jacp)
        self.attached = None
        self.relative_p = self.relative_R = None
        self.completed = []
        self.report = {"mode": "kinematic_arm_and_ideal_suction", "config": asdict(self.cfg),
                       "mujoco": mj.__version__, "phases": [], "placements": [], "success": False}
        self.steps_per_frame = max(1, round(self.cfg.frame_dt / model.opt.timestep))
        self.frame_dt = self.steps_per_frame * model.opt.timestep
        self.next_frame = time.monotonic()
        if initialize:
            k = self.optional_id("key", "home")
            if k >= 0:
                # Change only this robot and its parts, not unrelated Orca actors.
                self.d.qpos[self.qa] = model.key_qpos[k, self.qa]
            else:
                self.d.qpos[self.qa] = [-1.8, -1.4, 1.2, -1.4, -1.57, -.23]
            self.d.qvel[self.va] = 0
            self.d.ctrl[self.aids] = self.d.qpos[self.qa]
        mj.mj_forward(model, self.d)
        self.base_pos = self.d.xpos[self.base].copy()
        self.validate_bins()

    def optional_id(self, kind, name):
        obj = getattr(mj.mjtObj, "mjOBJ_"+kind.upper())
        requested = self.prefix + name
        exact = mj.mj_name2id(self.m, obj, requested)
        if exact >= 0:
            return exact
        count = getattr(self.m, {"body":"nbody","joint":"njnt","site":"nsite",
                                "geom":"ngeom","actuator":"nu","key":"nkey","equality":"neq"}[kind])
        # OrcaStudio adds a scene namespace and appends a UUID to geom names.
        # Match complete components so cell_1 cannot resolve to cell_10.
        uuid_tail = r"(?:_[0-9A-F]{8}(?:_[0-9A-F]{4}){3}_[0-9A-F]{12})?"
        pattern = re.compile(
            r"(?:^|[_/:])" + re.escape(requested) + uuid_tail + r"$",
            re.IGNORECASE,
        )
        hits = [
            i for i in range(count)
            if pattern.search(mj.mj_id2name(self.m, obj, i) or "")
        ]
        if len(hits) > 1:
            candidates = [mj.mj_id2name(self.m, obj, i) for i in hits]
            raise MotionError(
                f"Ambiguous {kind} {requested}: {candidates}; provide --prefix."
            )
        return hits[0] if hits else -1

    def id(self, kind, name):
        i = self.optional_id(kind, name)
        if i < 0:
            raise MotionError(f"Missing {kind}: {self.prefix+name}")
        return i

    def validate_bins(self):
        for b in set(DESTINATIONS):
            gid = self.id("geom", b+"_bottom")
            R = self.d.geom_xmat[gid].reshape(3,3)
            if not np.allclose(R, np.eye(3), atol=1e-5):
                raise MotionError("This scene adapter requires unrotated ground bins.")
        for n,b,off in zip(PARTS,DESTINATIONS,OFFSETS):
            half = self.half_extents(n)
            left = self.id("geom", b+"_wall_left")
            right = self.id("geom", b+"_wall_right")
            front = self.id("geom", b+"_wall_front")
            back = self.id("geom", b+"_wall_back")
            center = self.d.xpos[self.id("body",b)][:2] + off
            low = np.array([self.d.geom_xpos[left,0]+self.m.geom_size[left,0],
                            self.d.geom_xpos[front,1]+self.m.geom_size[front,1]])
            high = np.array([self.d.geom_xpos[right,0]-self.m.geom_size[right,0],
                             self.d.geom_xpos[back,1]-self.m.geom_size[back,1]])
            if np.any(center-half[:2] < low+.005) or np.any(center+half[:2] > high-.005):
                raise MotionError(f"{n}: full footprint will not fit inside {b}.")

    def half_extents(self,n):
        g=self.part_geoms[n];s=self.m.geom_size[g]
        if self.m.geom_type[g]==mj.mjtGeom.mjGEOM_CYLINDER:
            return np.array([s[0],s[0],s[1]])
        if self.m.geom_type[g]!=mj.mjtGeom.mjGEOM_BOX:
            raise MotionError(f"Unsupported part geometry: {n}")
        return s.copy()

    @staticmethod
    def rotation_error(current, target=DOWN):
        quat = np.empty(4)
        mj.mju_mat2Quat(quat, (target @ current.T).ravel())
        if quat[0] < 0:
            quat *= -1
        err = np.empty(3)
        mj.mju_quat2Vel(err, quat, 1.)
        return err

    def solve(self, target, seed):
        """
        6D DLS IK.

        The old version required roughly:
            position < 0.15 mm
            rotation < 0.034 deg

        That is unnecessarily strict for the OrcaStudio imported scene
        and caused valid poses such as 0.4 mm / 0.10 deg to be reported
        as unreachable.

        This version:
        1. keeps the best solution seen during iteration;
        2. allows an early high-precision exit;
        3. accepts the best solution if it satisfies the Controller
           configured Cartesian tolerances.
        """

        target = np.asarray(
            target,
            dtype=float
        )

        if (
            target.shape != (3,)
            or not np.isfinite(target).all()
        ):
            raise MotionError(
                "Invalid Cartesian target."
            )

        self.ik.qpos[:] = self.d.qpos
        self.ik.qpos[self.qa] = seed

        best_q = self.ik.qpos[
            self.qa
        ].copy()

        best_pos_err = float("inf")
        best_rot_err = float("inf")
        best_score = float("inf")

        # High precision target used only for early exit.
        precise_pos_tol = 0.0005
        precise_rot_tol = math.radians(0.15)

        # Final acceptable tolerance.
        #
        # Your Config already uses approximately:
        # position_tolerance = 1.5 mm
        # rotation_tolerance = 0.008 rad ~= 0.46 deg
        final_pos_tol = max(
            float(self.cfg.position_tolerance),
            0.0015
        )

        final_rot_tol = max(
            float(self.cfg.rotation_tolerance),
            math.radians(0.30)
        )

        for _ in range(220):

            mj.mj_forward(
                self.m,
                self.ik
            )

            ep = (
                target
                - self.ik.site_xpos[self.sid]
            )

            er = self.rotation_error(
                self.ik.site_xmat[
                    self.sid
                ].reshape(3,3)
            )

            pos_err = float(
                np.linalg.norm(ep)
            )

            rot_err = float(
                np.linalg.norm(er)
            )

            # Keep the best result rather than blindly using
            # the final iteration.
            score = (
                pos_err
                + 0.15 * rot_err
            )

            if score < best_score:

                best_score = score
                best_pos_err = pos_err
                best_rot_err = rot_err

                best_q = self.ik.qpos[
                    self.qa
                ].copy()

            # Precise early exit.
            if (
                pos_err <= precise_pos_tol
                and
                rot_err <= precise_rot_tol
            ):
                return self.ik.qpos[
                    self.qa
                ].copy()

            self.jacp[:] = 0
            self.jacr[:] = 0

            mj.mj_jacSite(
                self.m,
                self.ik,
                self.jacp,
                self.jacr,
                self.sid
            )

            J = np.vstack([
                self.jacp[:,self.va],
                .30 * self.jacr[:,self.va]
            ])

            error = np.r_[
                ep,
                .30 * er
            ]

            dq = (
                J.T
                @ np.linalg.solve(
                    J @ J.T
                    + 1e-4 * np.eye(6),
                    error
                )
            )

            dq = np.clip(
                dq,
                -.10,
                .10
            )

            self.ik.qpos[
                self.qa
            ] += dq

            for a,j in zip(
                self.qa,
                self.jids
            ):

                if self.m.jnt_limited[j]:

                    self.ik.qpos[a] = np.clip(
                        self.ik.qpos[a],
                        *self.m.jnt_range[j]
                    )

        # ----------------------------------------------------
        # IMPORTANT:
        # If the strict precise criterion was not reached,
        # accept the BEST pose if it is still within practical
        # project tolerances.
        # ----------------------------------------------------

        if (
            best_pos_err <= final_pos_tol
            and
            best_rot_err <= final_rot_tol
        ):

            print(
                "  IK accepted with practical tolerance: "
                f"position={best_pos_err*1000:.2f} mm, "
                f"rotation={math.degrees(best_rot_err):.3f} deg",
                flush=True
            )

            return best_q

        raise MotionError(
            f"Unreachable pose {np.round(target,3)}: "
            f"best_position={best_pos_err*1000:.1f} mm, "
            f"best_rotation={math.degrees(best_rot_err):.2f} deg"
        )

    def check_collision(self, label, *, payload=False):
        for c in self.d.contact:
            if c.dist >= -self.cfg.collision_tolerance:
                continue
            a,b=int(self.m.geom_bodyid[c.geom1]),int(self.m.geom_bodyid[c.geom2])
            arm_hit=(a in self.robot)!=(b in self.robot)
            if self.attached and {a,b} <= self.robot | {self.part_ids[self.attached]}:
                arm_hit=False
            payload_hit=payload and self.attached and (
                (a==self.part_ids[self.attached] and b not in self.robot)
                or (b==self.part_ids[self.attached] and a not in self.robot))
            self_hit = a in self.robot and b in self.robot and a != b and int(self.m.body_parentid[a]) != b and int(self.m.body_parentid[b]) != a
            if arm_hit or payload_hit or self_hit:
                g1=mj.mj_id2name(self.m,mj.mjtObj.mjOBJ_GEOM,c.geom1) or str(c.geom1)
                g2=mj.mj_id2name(self.m,mj.mjtObj.mjOBJ_GEOM,c.geom2) or str(c.geom2)
                raise MotionError(f"{label}: collision {g1} / {g2}, depth={-c.dist*1000:.2f} mm")

    def lock_payload(self):
        if self.attached is None:
            return
        R=self.d.site_xmat[self.sid].reshape(3,3)
        qa,va=self.free[self.attached]
        self.d.qpos[qa:qa+3]=self.d.site_xpos[self.sid]+R@self.relative_p
        quat=np.empty(4)
        mj.mju_mat2Quat(quat,(R@self.relative_R).ravel())
        self.d.qpos[qa+3:qa+7]=quat
        self.d.qvel[va:va+6]=0
        mj.mj_forward(self.m,self.d)

    def tick(self, q, label="", payload=False):
        # Exact arm projection is intentional in this kinematic demonstration.
        for _ in range(self.steps_per_frame):
            self.d.qpos[self.qa]=q
            self.d.qvel[self.va]=0
            self.d.ctrl[self.aids]=q
            mj.mj_forward(self.m,self.d)
            self.lock_payload()
            self.check_collision(label,payload=payload)
            mj.mj_step(self.m,self.d)
            self.d.qpos[self.qa]=q
            self.d.qvel[self.va]=0
            mj.mj_forward(self.m,self.d)
            self.lock_payload()
            if not np.isfinite(self.d.qpos).all() or not np.isfinite(self.d.qvel).all():
                raise MotionError("Non-finite simulation state.")
        if self.on_frame:
            self.on_frame(self)
        if self.cfg.realtime:
            self.next_frame += self.frame_dt
            wait=self.next_frame-time.monotonic()
            if wait>0:
                time.sleep(wait)
            elif wait < -.2:
                self.next_frame=time.monotonic()

    @staticmethod
    def min_jerk(u):
        """Quintic timing: zero velocity and acceleration at both ends."""
        return u*u*u*(10.0+u*(-15.0+6.0*u))

    def _finish_motion(self,target,label):
        err=float(np.linalg.norm(target-self.d.site_xpos[self.sid]))
        angle=float(np.linalg.norm(self.rotation_error(self.d.site_xmat[self.sid].reshape(3,3))))
        if err>self.cfg.position_tolerance or angle>self.cfg.rotation_tolerance:
            raise MotionError(f"{label}: pose tracking failed ({err} m, {angle} rad)")
        self.report["phases"].append({"label":label,"position_error_mm":err*1000,
                                     "angle_error_deg":math.degrees(angle)})
        print(f"  {label}: {err*1000:.2f} mm, {math.degrees(angle):.3f} deg",flush=True)

    def move(self,target,label,payload=False):
        target=np.asarray(target,dtype=float)
        start=self.d.site_xpos[self.sid].copy()
        q=self.d.qpos[self.qa].copy()
        # Plan before changing live state. One minimum-jerk timing law is
        # applied to the whole segment, so internal IK points never stop.
        count=max(4,math.ceil(1.5*np.linalg.norm(target-start)/(self.cfg.cartesian_speed*self.frame_dt)))
        trajectory=[]
        for i in range(1,count+1):
            s=self.min_jerk(i/count)
            qnext=self.solve(start+s*(target-start),q)
            n=max(1,math.ceil(np.max(np.abs(qnext-q))/(self.cfg.joint_speed*self.frame_dt)))
            trajectory.extend(q+(qnext-q)*j/n for j in range(1,n+1))
            q=qnext
        for q in trajectory:
            self.tick(q,label,payload)
        self._finish_motion(target,label)


    def move_physics(
        self,
        target,
        label,
        payload=False,
        max_seconds=2.5,
    ):
        """
        Dynamic Cartesian motion for teleoperation.

        IK provides only the desired joint configuration.
        The actual robot motion is generated by MuJoCo actuator dynamics.

        q_target -> joint position servo -> mj_step()
        """

        target=np.asarray(
            target,
            dtype=float
        )

        if (
            target.shape != (3,)
            or not np.isfinite(target).all()
        ):
            raise MotionError(
                "Invalid physics Cartesian target."
            )

        # --------------------------------------------------
        # IK target
        # --------------------------------------------------
        qgoal=self.solve(
            target,
            self.d.qpos[self.qa].copy()
        )

        # Position-servo control target.
        self.d.ctrl[self.aids]=qgoal

        dt=float(
            self.m.opt.timestep
        )

        if dt <= 0:
            raise MotionError(
                "Invalid MuJoCo timestep."
            )

        max_steps=max(
            1,
            int(max_seconds/dt)
        )

        render_every=max(
            1,
            int(round(self.frame_dt/dt))
        )

        site_error=float("inf")
        joint_error=float("inf")

        for step in range(max_steps):

            # ----------------------------------------------
            # TRUE physics integration.
            #
            # Do NOT overwrite:
            #   qpos
            #   qvel
            #
            # MuJoCo actuators move the robot.
            # ----------------------------------------------
            mj.mj_step(
                self.m,
                self.d
            )

            # Ideal vacuum constraint only while holding.
            if self.attached is not None:
                self.lock_payload()

            # ----------------------------------------------
            # Collision checking.
            # Physical contact is already solved by MuJoCo;
            # this layer prevents continued penetration.
            # ----------------------------------------------
            try:
                self.check_collision(
                    label,
                    payload=payload
                )

            except MotionError:

                # Stop commanding further motion.
                self.d.ctrl[
                    self.aids
                ]=self.d.qpos[
                    self.qa
                ].copy()

                if self.on_frame:
                    self.on_frame(self)

                raise

            site_error=float(
                np.linalg.norm(
                    target
                    - self.d.site_xpos[self.sid]
                )
            )

            joint_error=float(
                np.max(
                    np.abs(
                        qgoal
                        - self.d.qpos[self.qa]
                    )
                )
            )

            joint_speed=float(
                np.max(
                    np.abs(
                        self.d.qvel[self.va]
                    )
                )
            )

            if (
                self.on_frame
                and step % render_every == 0
            ):
                self.on_frame(self)

            # Cartesian AND joint-space convergence.
            if (
                site_error < .0020
                and joint_error < .008
                and joint_speed < .08
            ):
                break

        # Keep holding the final desired joint configuration.
        self.d.ctrl[
            self.aids
        ]=qgoal

        if self.on_frame:
            self.on_frame(self)

        site_error=float(
            np.linalg.norm(
                target
                - self.d.site_xpos[self.sid]
            )
        )

        joint_error=float(
            np.max(
                np.abs(
                    qgoal
                    - self.d.qpos[self.qa]
                )
            )
        )

        angle=float(
            np.linalg.norm(
                self.rotation_error(
                    self.d.site_xmat[
                        self.sid
                    ].reshape(3,3)
                )
            )
        )

        print(
            f"  {label}: "
            f"xyz={site_error*1000:.2f} mm, "
            f"joint={math.degrees(joint_error):.3f} deg, "
            f"rot={math.degrees(angle):.3f} deg "
            f"[physics]",
            flush=True
        )

        # Physical mode can retain a small steady-state error,
        # but anything beyond 6 mm indicates a real failure.
        if site_error > .006:
            raise MotionError(
                f"{label}: physics tracking error "
                f"{site_error*1000:.1f} mm"
            )

    def move_arc(self,center,radius,t0,t1,z,label,payload=False):
        """Follow one continuous arc; do not stop at intermediate angles."""
        center=np.asarray(center,dtype=float)
        delta=t1-t0
        q=self.d.qpos[self.qa].copy()
        count=max(8,math.ceil(1.5*abs(delta)*radius/
                              (self.cfg.cartesian_speed*self.frame_dt)))
        trajectory=[]
        target=None
        for i in range(1,count+1):
            s=self.min_jerk(i/count)
            angle=t0+s*delta
            target=np.r_[center+radius*np.array([math.cos(angle),math.sin(angle)]),z]
            qnext=self.solve(target,q)
            n=max(1,math.ceil(np.max(np.abs(qnext-q))/(self.cfg.joint_speed*self.frame_dt)))
            trajectory.extend(q+(qnext-q)*j/n for j in range(1,n+1))
            q=qnext
        for q in trajectory:
            self.tick(q,label,payload)
        self._finish_motion(target,label)

    def transfer(self,target_xy,z,label):
        """
        Collision-safe transfer.

        High-level approach:
            retract -> arc -> extend

        Payload-to-bin transfer:
            retract high
            -> arc high
            -> descend on transit radius to a geometry-safe clearance height
            -> cross the bin wall at that safe height
            -> descend vertically inside the bin

        This avoids both:
          1. unreachable far-away XY targets at travel_z
          2. collisions caused by crossing bin walls at low height
        """
        xy=np.asarray(target_xy,dtype=float)
        start=self.d.site_xpos[self.sid].copy()
        b=self.base_pos[:2]

        v0=start[:2]-b
        v1=xy-b

        t0=math.atan2(v0[1],v0[0])
        t1=math.atan2(v1[1],v1[0])

        delta=(t1-t0+math.pi)%(2*math.pi)-math.pi

        r=self.cfg.transit_radius
        # Dynamic transit height.
        #
        # The caller has already moved the TCP to a reachable safe height.
        # Do NOT force every transfer back to travel_z=1.0 m.
        #
        # Examples:
        #   approach: current_z ~= 0.912, target_z ~= 0.912
        #             -> high ~= 0.912
        #
        #   carry:    current_z ~= 0.92, target_z ~= 0.36
        #             -> high ~= 0.92
        current_z = float(self.d.site_xpos[self.sid,2])
        high = min(
            self.cfg.travel_z,
            max(
                current_z,
                float(z)
            )
        )
        
        print(
            f"  {label}/dynamic-high: "
            f"current_z={current_z:.3f}, "
            f"target_z={float(z):.3f}, "
            f"high={high:.3f} m",
            flush=True
        )
        radial0=np.r_[
            b+r*np.array([math.cos(t0),math.sin(t0)]),
            high
        ]

        radial1_high=np.r_[
            b+r*np.array([math.cos(t1),math.sin(t1)]),
            high
        ]

        # ----------------------------------------------------
        # Step 1: retract to safe radius at high Z.
        # ----------------------------------------------------
        self.move(
            radial0,
            label+"/retract",
            self.attached is not None
        )

        # ----------------------------------------------------
        # Step 2: rotate around the pedestal at high Z.
        # ----------------------------------------------------
        self.move_arc(
            b,
            r,
            t0,
            t0+delta,
            high,
            label+"/arc",
            self.attached is not None
        )

        # ----------------------------------------------------
        # Approach to a new part:
        # target Z == travel_z, so no bin-wall problem exists.
        # ----------------------------------------------------
        if not self.attached or z >= high-1e-9:
            self.move(
                np.r_[
                    b+r*np.array([math.cos(t1),math.sin(t1)]),
                    z
                ],
                label+"/height",
                self.attached is not None
            )

            self.move(
                np.r_[xy,z],
                label+"/extend",
                self.attached is not None
            )
            return

        # ====================================================
        # From here onward we are carrying a payload to a bin.
        # ====================================================

        # Find which destination bin this XY belongs to.
        bin_names=sorted(set(DESTINATIONS))

        bin_name=min(
            bin_names,
            key=lambda name: np.linalg.norm(
                xy-self.d.xpos[self.id("body",name)][:2]
            )
        )

        # ----------------------------------------------------
        # Find the actual top height of the four bin walls.
        # ----------------------------------------------------
        wall_ids=[
            self.id("geom",bin_name+"_wall_left"),
            self.id("geom",bin_name+"_wall_right"),
            self.id("geom",bin_name+"_wall_front"),
            self.id("geom",bin_name+"_wall_back"),
        ]

        wall_top=max(
            float(
                self.d.geom_xpos[g,2]
                + self.m.geom_size[g,2]
            )
            for g in wall_ids
        )

        # ----------------------------------------------------
        # Determine the lowest point of the payload relative
        # to the suction TCP.
        #
        # We use the current real MuJoCo geometry instead of
        # guessing from a fixed constant.
        # ----------------------------------------------------
        part=self.attached
        pg=self.part_geoms[part]

        R=self.d.geom_xmat[pg].reshape(3,3)
        ext=np.abs(R)@self.half_extents(part)

        payload_bottom=float(
            self.d.geom_xpos[pg,2]-ext[2]
        )

        tcp_z=float(
            self.d.site_xpos[self.sid,2]
        )

        bottom_offset=payload_bottom-tcp_z

        # 20 mm safety clearance above the highest bin wall.
        margin=.020

        clearance_z=max(
            float(z),
            wall_top-bottom_offset+margin
        )

        print(
            f"  {label}/clearance-plan: "
            f"bin={bin_name}, "
            f"wall_top={wall_top:.3f} m, "
            f"payload_bottom_offset={bottom_offset:.3f} m, "
            f"clearance_z={clearance_z:.3f} m",
            flush=True
        )

        if clearance_z >= high:
            raise MotionError(
                f"{label}: required bin clearance "
                f"{clearance_z:.3f} m >= travel_z "
                f"{high:.3f} m"
            )

        radial_clearance=np.r_[
            b+r*np.array([math.cos(t1),math.sin(t1)]),
            clearance_z
        ]

        # ----------------------------------------------------
        # Step 3:
        # Lower while still close to the robot.
        #
        # This avoids attempting the unreachable:
        #
        #     target XY + Z=1.00 m
        #
        # ----------------------------------------------------
        self.move(
            radial_clearance,
            label+"/clearance_height",
            True
        )

        # ----------------------------------------------------
        # Step 4:
        # Cross the bin wall only when the entire payload
        # bottom is safely above the wall.
        # ----------------------------------------------------
        self.move(
            np.r_[xy,clearance_z],
            label+"/extend_clearance",
            True
        )

        # ----------------------------------------------------
        # Step 5:
        # Now that XY is inside the bin, descend vertically.
        # ----------------------------------------------------
        self.move(
            np.r_[xy,z],
            label+"/descend",
            True
        )

    def enable_teleop_contacts(self):
        """
        Enable physical contact for the terminal suction tool and
        all detachable parts.

        Automatic disassembly does not call this method, so the
        already working automatic pipeline is not changed.
        """

        if not self.tool_geoms:
            raise MotionError(
                "No end-effector geoms found below suction_tcp body."
            )

        for gid in self.tool_geoms:
            self.m.geom_contype[gid] = 1
            self.m.geom_conaffinity[gid] = 1

        for name in PARTS:
            gid = self.part_geoms[name]
            self.m.geom_contype[gid] = 1
            self.m.geom_conaffinity[gid] = 1

        mj.mj_forward(self.m,self.d)


    def tool_contact_with_part(self,n):
        """
        Return the first physical MuJoCo contact between the suction
        tool subtree and part n.

        Returns:
            (True, contact)  if touching
            (False, None)    otherwise
        """

        target_body = self.part_ids[n]

        for con in self.d.contact:

            b1 = int(
                self.m.geom_bodyid[con.geom1]
            )

            b2 = int(
                self.m.geom_bodyid[con.geom2]
            )

            hit = (
                (
                    b1 in self.tool_bodies
                    and b2 == target_body
                )
                or
                (
                    b2 in self.tool_bodies
                    and b1 == target_body
                )
            )

            if hit:
                return True, con

        return False, None


    def attach_from_contact(self,n):
        """
        Engage the ideal vacuum constraint only after REAL physical
        tool-part contact has been detected.

        Unlike attach(), this does not require the imported
        suction_tcp and *_top_site positions to coincide.
        """

        if self.attached is not None:
            raise MotionError(
                "Already holding a part."
            )

        touching, con = self.tool_contact_with_part(n)

        if not touching:
            raise MotionError(
                f"{n}: vacuum rejected because "
                f"no physical tool-part contact exists."
            )

        # Require approximately downward tool orientation.
        R = self.d.site_xmat[
            self.sid
        ].reshape(3,3)

        axis = R[:,1]

        angle = math.acos(
            float(
                np.clip(
                    axis @ [0,0,-1],
                    -1,
                    1
                )
            )
        )

        if angle > math.radians(8.0):
            raise MotionError(
                f"{n}: tool angle too large "
                f"({math.degrees(angle):.2f} deg)"
            )

        fixture = self.optional_id(
            "equality",
            "fixture_"+n
        )

        if fixture >= 0:
            self.d.eq_active[fixture] = 0

        tcp = self.d.site_xpos[
            self.sid
        ].copy()

        bid = self.part_ids[n]

        self.relative_p = (
            R.T
            @ (
                self.d.xpos[bid]
                - tcp
            )
        )

        self.relative_R = (
            R.T
            @ self.d.xmat[
                bid
            ].reshape(3,3)
        )

        self.attached = n

        self.lock_payload()

        depth = float(
            max(0.0,-con.dist)
        )

        print(
            f"  VACUUM CONTACT: {n}, "
            f"penetration={depth*1000:.2f} mm",
            flush=True
        )


    def attach(self,n):
        tcp=self.d.site_xpos[self.sid]
        surface=self.d.site_xpos[self.part_sites[n]]
        dist=float(np.linalg.norm(surface-tcp))
        axis=self.d.site_xmat[self.sid].reshape(3,3)[:,1]
        angle=math.acos(float(np.clip(axis@[0,0,-1],-1,1)))
        if dist>self.cfg.grasp_tolerance or angle>self.cfg.angle_tolerance:
            raise MotionError(f"{n}: grasp rejected ({dist*1000:.2f} mm / {math.degrees(angle):.2f} deg)")
        if self.attached:
            raise MotionError("Already holding a part.")
        fixture=self.optional_id("equality","fixture_"+n)
        if fixture>=0:
            self.d.eq_active[fixture]=0
        R=self.d.site_xmat[self.sid].reshape(3,3)
        bid=self.part_ids[n]
        self.relative_p=R.T@(self.d.xpos[bid]-tcp)
        self.relative_R=R.T@self.d.xmat[bid].reshape(3,3)
        self.attached=n
        self.lock_payload()

    def wait(self,seconds,label):
        for _ in range(math.ceil(seconds/self.frame_dt)):
            self.tick(self.d.qpos[self.qa].copy(),label)

    def verify_placement(self,n,b,expected):
        g=self.part_geoms[n]

        R=self.d.geom_xmat[g].reshape(3,3)
        ext=np.abs(R)@self.half_extents(n)
        center=self.d.geom_xpos[g]

        left=self.id("geom",b+"_wall_left")
        right=self.id("geom",b+"_wall_right")
        # 两个蓝色电芯箱共用一块中间挡板。
        # 对 bin_cell_2 来说，bin_cell_wall_back 就是它的前挡板。
        if b == "bin_cell_2":
            front=self.id("geom","bin_cell_wall_back")
        else:
            front=self.id("geom",b+"_wall_front")
        back=self.id("geom",b+"_wall_back")

        low=np.array([
            self.d.geom_xpos[left,0]+self.m.geom_size[left,0],
            self.d.geom_xpos[front,1]+self.m.geom_size[front,1]
        ])

        high=np.array([
            self.d.geom_xpos[right,0]-self.m.geom_size[right,0],
            self.d.geom_xpos[back,1]-self.m.geom_size[back,1]
        ])

        floor=self.id("geom",b+"_bottom")
        floor_z=(
            self.d.geom_xpos[floor,2]
            + self.m.geom_size[floor,2]
        )

        _,va=self.free[n]

        # MuJoCo freejoint qvel:
        # [vx, vy, vz, wx, wy, wz]
        #
        # Do NOT mix linear velocity (m/s) and angular velocity (rad/s)
        # into one Euclidean norm.
        vel=self.d.qvel[va:va+6]

        linear_speed=float(
            np.linalg.norm(vel[:3])
        )

        angular_speed=float(
            np.linalg.norm(vel[3:])
        )

        inside_xy=bool(
            np.all(center[:2]-ext[:2] >= low-.002)
            and
            np.all(center[:2]+ext[:2] <= high+.002)
        )

        bottom_error=float(
            abs(center[2]-ext[2]-floor_z)
        )

        on_floor=bottom_error < .006

        # MuJoCo cylinder-floor contact can retain small numerical
        # sliding / spinning even when the cell is correctly placed.
        #
        # For cells, placement correctness is determined primarily by:
        #   - inside the bin
        #   - resting on the floor
        #   - remaining near the intended XY
        #   - remaining upright
        #
        # Velocity thresholds therefore reject only obvious instability.
        xy_error=float(
            np.linalg.norm(center[:2]-np.asarray(expected,dtype=float))
        )

        if n.startswith("battery_cell_"):
            upright=float(abs(R[2,2])) > math.cos(math.radians(12.0))

            stable=(
                linear_speed < .08
                and angular_speed < 1.50
                and xy_error < .015
                and upright
            )
        else:
            upright=True

            stable=(
                linear_speed < .03
                and angular_speed < .50
            )

        ok=bool(
            inside_xy
            and on_floor
            and stable
        )

        record={
            "part":n,
            "bin":b,
            "center":center.tolist(),
            "bottom_z":float(center[2]-ext[2]),
            "floor_z":float(floor_z),
            "bottom_error_mm":bottom_error*1000.0,
            "expected_xy":list(expected),
            "linear_speed_mps":linear_speed,
            "angular_speed_radps":angular_speed,
            "xy_error_mm":xy_error*1000.0,
            "upright":upright,
            "inside_xy":inside_xy,
            "on_floor":on_floor,
            "stable":stable,
            "success":ok
        }

        self.report["placements"].append(record)

        print(
            f"  verify {n}: "
            f"linear={linear_speed:.4f} m/s, "
            f"angular={angular_speed:.4f} rad/s, "
            f"bottom_error={bottom_error*1000:.2f} mm, "
            f"xy_error={xy_error*1000:.2f} mm, "
            f"upright={upright}, "
            f"inside={inside_xy}",
            flush=True
        )

        if not ok:
            raise MotionError(
                f"{n}: did not settle fully inside {b}: {record}"
            )

        return record

    def pick_place(self,n,b,off):
        print(f"[{PARTS.index(n)+1}/7] {n} -> {b}",flush=True)
        surface=self.d.site_xpos[self.part_sites[n]].copy()

        # Diagnose whether the imported *_top_site is really on the
        # physical top of the part. Orca/USD conversion can leave a site
        # inside the mesh, which makes the TCP press through the object.
        g = self.part_geoms[n]
        Rg = self.d.geom_xmat[g].reshape(3,3)
        ext = np.abs(Rg) @ self.half_extents(n)
        geom_top_z = float(self.d.geom_xpos[g,2] + ext[2])

        print(
            f"  DEBUG {n}: "
            f"site_z={surface[2]:.4f}, "
            f"geom_center_z={self.d.geom_xpos[g,2]:.4f}, "
            f"geom_top_z={geom_top_z:.4f}, "
            f"site_minus_top={(surface[2]-geom_top_z)*1000:.1f} mm",
            flush=True
        )

        # Do not blindly trust the imported top_site Z coordinate.
        # Use the higher of top_site and collision-geometry top.
        surface[2] = max(surface[2], geom_top_z)

        # Keep the suction TCP slightly above the actual top surface.
        pick=surface+np.array([0,0,self.cfg.grasp_gap])

        print(
            f"  DEBUG {n}: corrected_pick_z={pick[2]:.4f}",
            flush=True
        )

        cur=self.d.site_xpos[self.sid].copy()

        # Adaptive high pose:
        # do not force the robot to Z=1.0 m when the same XY
        # is close to the edge of the UR5e workspace.
        #
        # Keep roughly 120 mm above the current part while
        # limiting the high pose to 0.94 m.
        safe_high_z = min(
            self.cfg.travel_z,
            max(
                pick[2] + .12,
                .90
            ),
            .94
        )

        print(
            f"  DEBUG {n}: safe_high_z={safe_high_z:.3f} m",
            flush=True
        )

        self.move(
            np.r_[cur[:2], safe_high_z],
            "prepare"
        )
        # Returning from a ground bin must retract at low height first.
        # ----------------------------------------------------
        # Part-specific approach height.
        #
        # Cells are farther from the UR5e base.  Forcing the TCP to
        # travel_z=1.00 m directly above cell_3 / cell_4 puts the arm
        # close to its workspace boundary while maintaining the
        # downward suction orientation.
        #
        # Approach cells at a lower but still safe height instead.
        # ----------------------------------------------------
        approach_z = safe_high_z

        print(
            f"  DEBUG {n}: approach_z={approach_z:.3f} m",
            flush=True
        )

        self.transfer(
            pick[:2],
            approach_z,
            "approach"
        )
        self.move(pick,"pick")
        self.attach(n)
        # The stacked parts can start with a small contact penetration after
        # Orca's USD conversion. Move straight up to separate them first.
        # Extraction must always move upward.
        clearance_lift = .100 if n == "battery_cover" else .030
        separation_z = pick[2] + clearance_lift

        # Never silently clamp an extraction target downward.
        if self.cfg.travel_z < separation_z:
            raise MotionError(
                f"{n}: travel_z={self.cfg.travel_z:.3f} m is below "
                f"required separation_z={separation_z:.3f} m"
            )
        self.move(np.r_[pick[:2],separation_z],"separate")
        self.check_collision("separate/clearance",payload=True)
        lift_z = min(
            self.cfg.travel_z,
            max(
                safe_high_z,
                separation_z + .03
            ),
            .94
        )

        print(
            f"  DEBUG {n}: lift_z={lift_z:.3f} m",
            flush=True
        )

        self.move(
            np.r_[pick[:2], lift_z],
            "lift",
            True
        )
        xy=self.d.xpos[self.id("body",b)][:2]+off
        self.transfer(xy,self.cfg.bin_hover_z,"carry")
        floor=self.id("geom",b+"_bottom")
        floor_z=self.d.geom_xpos[floor,2]+self.m.geom_size[floor,2]
        # Desired center + preserved center-to-TCP offset yields release TCP.
        # Release very close to the bin floor.
        # Tall cylindrical cells bounce significantly if dropped from 15 mm.
        if n.startswith("battery_cell_"):
            release_gap = .002
        else:
            release_gap = .005

        center=np.r_[
            xy,
            floor_z+self.half_extents(n)[2]+release_gap
        ]

        release=center-DOWN@self.relative_p
        self.move(release,"place",True)
        self.attached=None

        # Cylindrical cells need more time to settle after release.
        initial_settle = 3.0 if n.startswith("battery_cell_") else self.cfg.settle_seconds
        self.wait(initial_settle,"settle")

        _,va=self.free[n]

        for i in range(20):
            vel=self.d.qvel[va:va+6]

            linear_speed=float(
                np.linalg.norm(vel[:3])
            )

            angular_speed=float(
                np.linalg.norm(vel[3:])
            )

            print(
                f"  settle {n}: "
                f"linear={linear_speed:.4f} m/s, "
                f"angular={angular_speed:.4f} rad/s",
                flush=True
            )

            if (
                linear_speed < .03
                and angular_speed < .50
            ):
                break

            self.wait(.25,"settle/adaptive")

        self.verify_placement(n,b,xy)
        self.completed.append(n)
        self.move(np.r_[xy,self.cfg.bin_hover_z],"retreat")
        # Pull inward before rising: high TCP at outer bin radius is unreachable.
        v=xy-self.base_pos[:2];v*=self.cfg.transit_radius/np.linalg.norm(v)
        self.move(np.r_[self.base_pos[:2]+v,self.cfg.bin_hover_z],"return/retract")
        # Do not return to travel_z=1.0 m.
        # At the transit radius that high pose is near the UR5e workspace limit.
        return_z = min(
            self.cfg.travel_z,
            .90
        )

        print(
            f"  return/safe-height: {return_z:.3f} m",
            flush=True
        )

        self.move(
            np.r_[
                self.base_pos[:2] + v,
                return_z
            ],
            "return/lift"
        )

    def run(self,through=None):
        """Prefix-only testing preserves cover -> busbar -> cell dependencies."""
        sequence=list(zip(PARTS,DESTINATIONS,OFFSETS))
        if through is not None:
            sequence=sequence[:PARTS.index(through)+1]
        try:
            self.wait(.1,"initialize")
            for n,b,off in sequence:
                self.pick_place(n,b,off)
            # Recheck all placements after subsequent movements.
            for n,b,off in sequence:
                self.verify_placement(n,b,self.d.xpos[self.id("body",b)][:2]+off)
            self.report["success"]=True
        except (MotionError,KeyboardInterrupt) as exc:
            # Stop the entire sequence; keep held part attached. No high-altitude release.
            self.report["error"]=str(exc) or "Interrupted"
            raise
        finally:
            self.report["completed"]=self.completed.copy()
            self.report["holding"]=self.attached
            self.report["simulation_seconds"]=float(self.d.time)

    def save_report(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(self.report,ensure_ascii=False,indent=2),encoding="utf-8")
