#!/usr/bin/env python3
"""OrcaStudio/OrcaLab bridge using the installed OrcaGymLocal backend.

Physics runs in the SDK's local MuJoCo model; gym.render() synchronizes qpos
(including all seven freejoint parts) to the running Orca scene.
"""
import argparse
import asyncio
import os
from pathlib import Path
import sys
import tempfile
import time
import mujoco
import numpy as np
from disassembly import Controller, Config, MotionError, PARTS

# 每次修改相机弹窗逻辑就更新这个标记；启动时打印出来，用来确认
# 当前运行的进程加载的是最新代码（而不是改动前启动的旧进程）。
CAMERA_VIEW_VERSION = "v4-studio-frame"

# scene_two_pack.xml 中俯视拆解工位的相机。
DEFAULT_CAMERA = "station_camera"

# OrcaStudio 不会把 <camera> 导出给 SDK，因此找不到相机时用自由相机代替。
# 相机位姿优先取模型里 station_camera_mount body 的实时位置（OrcaStudio 的世界
# 坐标与本地 XML 不同，不能用固定常量）；DEFAULT_CAMERA_POSE 仅作最后兜底。
DEFAULT_CAMERA_POSE = (-0.55, 0.20, 1.73)

def top_down_camera(pose=DEFAULT_CAMERA_POSE):
    """按 (x, y, z) 构造一个朝正下方的自由相机。"""
    cam=mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:]=[pose[0],pose[1],0.0]
    cam.distance=float(pose[2])
    cam.azimuth=90.0
    cam.elevation=-90.0
    return cam

class CameraViewer:
    """把 MuJoCo 相机画面实时渲染到 OpenCV 弹窗。

    每帧调用 draw()：离屏渲染指定相机，再用 cv2.imshow 刷新窗口。
    """

    def __init__(self,model,camera,label=DEFAULT_CAMERA,width=480,height=360,
                 all_geom_groups=False,brighten=False):
        try:
            import cv2
        except ImportError as e:
            raise MotionError("Camera view needs opencv-python (pip install opencv-python).") from e
        self.cv2=cv2
        # 模型里的相机 id，或 MjvCamera 自由相机。
        self.camera=camera
        self.title=f"OrcaStudio camera: {label}"
        self.renderer=mujoco.Renderer(model,height=height,width=width)
        # OrcaStudio 导出的几何体可能都在 group 3/4（MuJoCo 默认不显示），
        # 全部打开以免渲染出一片空白。
        self.scene_option=None
        if all_geom_groups:
            self.scene_option=mujoco.MjvOption()
            mujoco.mjv_defaultOption(self.scene_option)
            self.scene_option.geomgroup[:]=1
        # 碰撞代理体没有任何材质/光照数据，默认渲染出来很暗；提高 scene 灯光亮度。
        self.brighten=brighten
        self._window=False
        self._disabled=False
        self.frames=0

    def draw(self,data):
        if self._disabled:
            return
        try:
            self.renderer.update_scene(data,camera=self.camera,scene_option=self.scene_option)
            if self.brighten:
                scene=self.renderer.scene
                for i in range(scene.nlight):
                    scene.lights[i].ambient[:]=0.45
                    scene.lights[i].diffuse[:]=1.0
            frame=self.renderer.render()
            self.frames+=1
            # 前 30 帧每帧都打印，之后每 50 帧一次：确保启动后立刻能看到
            # 弹窗是否在刷新（frame 编号持续增长 = 刷新正常）。
            if self.frames<=30 or self.frames % 50 == 0:
                print(f"[CAMERA VIEW] frame {self.frames}, t={float(data.time):.2f}s, "
                      f"mean={float(frame.mean()):.0f}",flush=True)
            if not self._window:
                self.cv2.namedWindow(self.title,self.cv2.WINDOW_NORMAL)
                self._window=True
            # MuJoCo 输出 RGB，cv2 需要 BGR。
            self.cv2.imshow(self.title,frame[:,:,::-1])
            self.cv2.waitKey(1)
        except Exception as e:
            print(f"[CAMERA VIEW] disabled after render error: {type(e).__name__}: {e}",file=sys.stderr)
            self._disabled=True

    def close(self):
        try:
            if self._window:
                self.cv2.destroyWindow(self.title)
        except Exception:
            pass
        try:
            self.renderer.close()
        except Exception:
            pass

def _joint_pairs(online_model,prefixes,local_model):
    """按（去掉命名空间前缀的）关节名把在线模型和本地模型配对。"""
    def norm(name):
        if not name:
            return name
        for pre in prefixes:
            if pre and name.startswith(pre):
                return name[len(pre):]
        return name
    local_by_name={}
    for j in range(local_model.njnt):
        local_by_name[norm(mujoco.mj_id2name(local_model,mujoco.mjtObj.mjOBJ_JOINT,j))]=j
    # mjtJoint 是 IntEnum，但 dict 用 np.int32 作键查不到（必须转成 int）。
    free=int(mujoco.mjtJoint.mjJNT_FREE)
    ball=int(mujoco.mjtJoint.mjJNT_BALL)
    pairs=[]
    for j in range(online_model.njnt):
        name=norm(mujoco.mj_id2name(online_model,mujoco.mjtObj.mjOBJ_JOINT,j))
        lj=local_by_name.get(name)
        if lj is None:
            continue
        t=int(online_model.jnt_type[j])
        qdim={free:7,ball:4}.get(t,1)
        vdim={free:6,ball:3}.get(t,1)
        pairs.append((
            int(online_model.jnt_qposadr[j]),int(online_model.jnt_dofadr[j]),int(qdim),int(vdim),t,
            int(local_model.jnt_qposadr[lj]),int(local_model.jnt_dofadr[lj]),
        ))
    return pairs


class MirrorCameraViewer:
    """在线模式下显示本地网格模型（镜像在线状态）。

    OrcaStudio 不把网格导出给 SDK（nmesh=0），也不导出 <camera>（ncam=0），
    所以在线模型只能渲染出碰撞代理方块。这里额外加载本地带网格的 XML
    （scene_two_pack.xml + ur5e_arm.xml），每帧把在线模型的关节/自由体状态按
    名字同步进来（两个世界只差一个固定平移），再用本地相机渲染，从而在线也
    能看到真实的 UR5e 机械臂跟随运动。
    """

    def __init__(self,online_data,local_model,local_data,camera,offset,pairs,label,
                 width=480,height=360,tmp_path=None):
        try:
            import cv2
        except ImportError as e:
            raise MotionError("Camera view needs opencv-python (pip install opencv-python).") from e
        self.cv2=cv2
        self.online_data=online_data
        self.lm=local_model
        self.ld=local_data
        # 本地模型里的相机 id（int）或自由相机（MjvCamera）。
        self.camera=camera
        self.offset=np.asarray(offset,dtype=float)
        self.pairs=pairs
        self.title=f"OrcaStudio camera: {label}"
        self.renderer=mujoco.Renderer(local_model,height=height,width=width)
        self.tmp_path=tmp_path
        self._window=False
        self._disabled=False
        self.frames=0

    def _sync_state(self):
        od=self.online_data
        ld=self.ld
        free=int(mujoco.mjtJoint.mjJNT_FREE)
        for oq,ov,qdim,vdim,jtype,lq,lv in self.pairs:
            ld.qpos[lq:lq+qdim]=od.qpos[oq:oq+qdim]
            if jtype==free:
                # 在线世界 = 本地世界 + offset，自由体位置换算回本地世界。
                ld.qpos[lq:lq+3]-=self.offset
            ld.qvel[lv:lv+vdim]=od.qvel[ov:ov+vdim]
        mujoco.mj_forward(self.lm,self.ld)

    def draw(self,data):
        if self._disabled:
            return
        try:
            self._sync_state()
            self.renderer.update_scene(self.ld,camera=self.camera)
            frame=self.renderer.render()
            self.frames+=1
            # 前 30 帧每帧都打印，之后每 50 帧一次，方便确认画面在刷新。
            if self.frames<=30 or self.frames%50==0:
                print(f"[CAMERA VIEW] frame {self.frames}, t={float(data.time):.2f}s, "
                      f"mean={float(frame.mean()):.0f} (mesh mirror)",flush=True)
            if not self._window:
                self.cv2.namedWindow(self.title,self.cv2.WINDOW_NORMAL)
                self._window=True
            # MuJoCo 输出 RGB，cv2 需要 BGR。
            self.cv2.imshow(self.title,frame[:,:,::-1])
            self.cv2.waitKey(1)
        except Exception as e:
            print(f"[CAMERA VIEW] disabled after render error: {type(e).__name__}: {e}",flush=True)
            self._disabled=True

    def close(self):
        try:
            if self._window:
                self.cv2.destroyWindow(self.title)
        except Exception:
            pass
        try:
            self.renderer.close()
        except Exception:
            pass
        if self.tmp_path is not None:
            try:
                Path(self.tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

class StudioFrameViewer:
    """显示 OrcaStudio 视口的真实画面（gRPC GetCameraFramePNG 抓帧）。

    OrcaStudio 不把 <camera> 和网格导出给 SDK，但引擎侧提供了抓帧接口
    ``GetCameraFramePNG(image_path)``：它把**当前 Studio 视口**渲染出的画面写成
    PNG。这里周期调用它，再把 PNG 显示到 OpenCV 弹窗，所以看到的就是 OrcaStudio
    里的真实画面（含网格、材质、光照）。

    注意事项：
    - 该接口没有相机名参数，抓到的是 Studio 当前激活的视口，而不是指定的
      ``station_camera``；因此不会去改动用户在 OrcaStudio 里的视角。
    - 它涉及 GPU 回读，OrcaGym 自身记录「与渲染流并发可能触发
      AttachmentReadback::CopyExecute 崩溃」。这里做了两件事降低风险：
      调用做了节流；并且抓帧与 gym.render() 都在同一个事件循环里顺序 await，
      不会出现两个 RPC 同时在飞。
    - 任何一次抓帧失败（引擎不支持 / 文件没写出 / PNG 读不出来）都会自动停用
      弹窗并提示改用 ``--camera-source mirror``。
    """

    def __init__(self,session,label=DEFAULT_CAMERA,interval=0.25):
        try:
            import cv2
        except ImportError as e:
            raise MotionError("Camera view needs opencv-python (pip install opencv-python).") from e
        self.cv2=cv2
        self.session=session
        self.title=f"OrcaStudio viewport: {label}"
        self.interval=float(interval)
        # 引擎按给定路径落盘；用正斜杠，避免 Orca 解析 Windows 反斜杠出问题。
        self.path=str(Path(tempfile.gettempdir())/f"orca_viewport_{os.getpid()}.png").replace("\\","/")
        self._last=-1e9
        self._window=False
        self._disabled=False
        self.frames=0

    def grab(self):
        """抓一帧，返回引擎写出的 PNG 路径（同步阻塞）。"""
        async def call():
            return await asyncio.wait_for(self.session.gym.get_frame_png(self.path),
                                          self.session.timeout)
        self.session.loop.run_until_complete(call())
        return self.path

    def draw(self,data):
        if self._disabled:
            return
        now=time.time()
        # 节流：抓帧是整帧 PNG 往返 + GPU 回读，不能每帧都做。
        if now-self._last < self.interval:
            return
        self._last=now
        try:
            self.grab()
            if not Path(self.path).exists():
                raise FileNotFoundError(f"engine did not write {self.path}")
            # cv2.imread 出来就是 BGR，正好是 imshow 需要的顺序。
            frame=self.cv2.imread(self.path)
            if frame is None:
                raise ValueError("engine returned an unreadable PNG")
            self.frames+=1
            if self.frames<=10 or self.frames % 25 == 0:
                print(f"[CAMERA VIEW] studio frame {self.frames}, t={float(data.time):.2f}s, "
                      f"shape={frame.shape}, mean={float(frame.mean()):.0f}",flush=True)
            if not self._window:
                self.cv2.namedWindow(self.title,self.cv2.WINDOW_NORMAL)
                self._window=True
            self.cv2.imshow(self.title,frame)
            self.cv2.waitKey(1)
        except Exception as e:
            self._disabled=True
            print(f"[CAMERA VIEW] studio frame grab failed -> popup disabled: "
                  f"{type(e).__name__}: {e}",flush=True)
            print("[CAMERA VIEW] tip: run with --camera-source mirror to render the "
                  "local mesh model instead.",flush=True)

    def close(self):
        try:
            if self._window:
                self.cv2.destroyWindow(self.title)
        except Exception:
            pass
        try:
            Path(self.path).unlink(missing_ok=True)
        except Exception:
            pass


class OrcaSession:
    def __init__(self,addr,offline_xml=None,timeout=15):
        try:
            import grpc
            from orca_gym import OrcaGymLocal
            from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub
        except ImportError as e:
            raise MotionError("Use an environment with orca-gym installed (e.g. conda activate orcalab).") from e
        self.loop=asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.channel=None
        self.timeout=timeout
        self.offline=offline_xml is not None
        self._viewer=None
        async def connect():
            step="initializing the offline simulation"
            try:
                if self.offline:
                    self.gym=OrcaGymLocal(None,skip_grpc_load=True,local_xml_path=str(offline_xml))
                    await self.gym.init_simulation(str(offline_xml))
                else:
                    step="connecting to the OrcaLab gRPC channel"
                    self.channel=grpc.aio.insecure_channel(addr,options=[
                        ("grpc.max_receive_message_length",1024**3),
                        ("grpc.max_send_message_length",1024**3)])
                    await asyncio.wait_for(self.channel.channel_ready(),timeout)
                    self.gym=OrcaGymLocal(GrpcServiceStub(self.channel))
                    step="downloading the model XML from OrcaStudio"
                    xml=await asyncio.wait_for(self.gym.load_model_xml(),120)
                    step="pausing the OrcaStudio simulation"
                    await asyncio.wait_for(self.gym.pause_simulation(),timeout)
                    step="initializing the simulation"
                    await self.gym.init_simulation(xml)
            except asyncio.TimeoutError as e:
                # 裸 TimeoutError 没有任何信息；这里补上失败发生在哪一步以及如何排查。
                raise MotionError(
                    f"OrcaStudio did not respond within the timeout while {step} "
                    f"(addr={addr!r}, timeout={timeout}s). Check that OrcaStudio is open with "
                    "scene_two_pack.xml loaded and the simulation RUNNING (press Run), and that "
                    "--addr points at the OrcaLab bridge."
                ) from e
        try:
            self.loop.run_until_complete(connect())
            self.model=self.gym._mjModel
            self.data=self.gym._mjData
            if not isinstance(self.model,mujoco.MjModel) or not isinstance(self.data,mujoco.MjData):
                raise MotionError("Unsupported Orca SDK: expected OrcaGymLocal MuJoCo backend.")
            self.last_frame=-1.
        except BaseException:
            self.close()
            raise

    def sync(self,controller,force=False):
        # 相机窗口优先刷新：直接拿当前本地模型状态渲染，不依赖时间节流，
        # 也不会被后面的 OrcaStudio RPC 拖住。即使 RPC 很慢或失败，
        # 弹窗也能持续更新。
        if self._viewer is not None:
            self._viewer.draw(self.data)
        # Controller frames are 20 ms (50 Hz). Send every new frame; the old
        # 1/30 threshold actually produced uneven 40 ms updates (25 Hz).
        if force or controller.d.time > self.last_frame + 1e-12:
            self.gym.update_data()
            if not self.offline:
                self.loop.run_until_complete(asyncio.wait_for(self.gym.render(),self.timeout))
            self.last_frame=float(controller.d.time)

    def enable_camera_view(self,controller,camera=DEFAULT_CAMERA,source="auto"):
        """打开实时相机弹窗。

        ``source`` 决定画面从哪里来：
        - ``studio``：在线时抓 OrcaStudio 视口的真实画面（gRPC GetCameraFramePNG）。
        - ``mirror``：用本地带网格模型做镜像渲染。
        - ``auto``（默认）：在线先试 studio，抓帧不可用则自动退回 mirror。

        离线运行时没有引擎可抓帧，直接走本地相机/镜像。相机不存在时不中断
        拆解流程。
        """
        print(f"[CAMERA VIEW] {CAMERA_VIEW_VERSION}: popup requested "
              f"(camera='{camera}', source={source}).",flush=True)
        if not self.offline and source in ("auto","studio"):
            studio=self._enable_studio_view(camera)
            if studio is not None:
                self._viewer=studio
                return studio
            if source=="studio":
                print("[CAMERA VIEW] studio capture unavailable; falling back to mesh mirror.",flush=True)
        camera_id=controller.optional_id("camera",camera)
        if camera_id >= 0:
            self._viewer=CameraViewer(self.model,camera_id,camera)
            return self._viewer
        # OrcaStudio 不把 <camera>（和网格）导出给 SDK：优先用本地网格模型做镜像渲染。
        mirror=self._enable_mirror_view(controller,camera)
        if mirror is not None:
            self._viewer=mirror
            return mirror
        if camera != DEFAULT_CAMERA:
            available=[mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_CAMERA,i)
                       for i in range(self.model.ncam)]
            print(
                f"[CAMERA VIEW] '{camera}' not found and no fallback pose; popup disabled. "
                f"Available cameras: {available or 'none'}.",
                flush=True,
            )
            return None
        print(
            f"[CAMERA VIEW] '{camera}' not exported by OrcaStudio (ncam={self.model.ncam}); "
            "using a built-in top-down free camera.",
            flush=True,
        )
        # OrcaStudio 的世界坐标与本地 XML 不同，必须从模型里取相机安装座的实时位置。
        pose=DEFAULT_CAMERA_POSE
        mount_id=controller.optional_id("body",f"{camera}_mount")
        if mount_id >= 0:
            mujoco.mj_forward(self.model,self.data)
            p=self.data.xpos[mount_id]
            pose=(float(p[0]),float(p[1]),float(p[2]))
            print(f"[CAMERA VIEW] pose from body '{camera}_mount': {np.round(pose,3).tolist()}",flush=True)
        else:
            print(f"[CAMERA VIEW] body '{camera}_mount' not found; using fallback pose {DEFAULT_CAMERA_POSE}.",flush=True)
        self._viewer=CameraViewer(self.model,top_down_camera(pose),f"{camera} (top-down)",
                                  all_geom_groups=True,brighten=True)
        # 诊断：几何体数量与世界包围盒，用于确认相机是否对着场景。
        xs=[self.data.geom_xpos[g] for g in range(self.model.ngeom)
            if self.model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_PLANE]
        bbox="n/a" if not xs else (
            f"{np.round(np.min(xs,0),2).tolist()} .. {np.round(np.max(xs,0),2).tolist()}")
        print(
            f"[CAMERA VIEW] diag: ngeom={self.model.ngeom}, nmesh={self.model.nmesh}, "
            f"geom bbox(world)={bbox}",
            flush=True,
        )
        return self._viewer

    def _enable_studio_view(self,camera):
        """在线时抓 OrcaStudio 视口的真实画面。

        先探测一次抓帧；引擎不支持 / 文件没写出就直接返回 None，让调用方退回
        镜像渲染（不留下一个空白弹窗）。
        """
        print("[CAMERA VIEW] grabbing the OrcaStudio viewport via GetCameraFramePNG "
              "(engine-side readback; throttled to ~4 Hz).",flush=True)
        try:
            viewer=StudioFrameViewer(self,camera)
            viewer.grab()
            if not Path(viewer.path).exists():
                raise FileNotFoundError(f"engine did not write {viewer.path}")
        except Exception as e:
            print(f"[CAMERA VIEW] studio capture probe failed: {type(e).__name__}: {e}",flush=True)
            return None
        print(f"[CAMERA VIEW] studio viewport capture enabled; PNG at {viewer.path}",flush=True)
        # 探测已经抓到一帧，避免立刻再抓一次。
        viewer._last=time.time()
        return viewer

    def _enable_mirror_view(self,controller,camera):
        """用本地带网格的 XML 构造镜像渲染（在线模式用）。

        若本地 scene/arm XML 不存在、加载失败或没有可配对的关节，则返回 None，
        由调用方退回到碰撞代理体的俯视视角。
        """
        here=Path(__file__).resolve().parent
        scene=here/"scene_two_pack.xml"
        arm=here/"ur5e_arm.xml"
        if not scene.exists():
            print(f"[CAMERA VIEW] mirror skipped: {scene} not found.",flush=True)
            return None
        tmp=None
        try:
            src=_offline_combined(scene,arm) if arm.exists() else scene
            tmp=src if src!=scene else None
            lmodel=mujoco.MjModel.from_xml_path(str(src))
        except Exception as e:
            print(f"[CAMERA VIEW] mirror model failed to load: {type(e).__name__}: {e}",flush=True)
            if tmp is not None:
                tmp.unlink(missing_ok=True)
            return None
        ldata=mujoco.MjData(lmodel)
        mujoco.mj_forward(lmodel,ldata)
        prefixes=(controller.prefix,controller.arm_prefix)
        pairs=_joint_pairs(self.model,prefixes,lmodel)
        if not pairs:
            print("[CAMERA VIEW] mirror skipped: no matching joints.",flush=True)
            if tmp is not None:
                tmp.unlink(missing_ok=True)
            return None
        # 世界偏移 = 在线安装座 - 本地安装座（两个世界只差一个纯平移）。
        offset=np.zeros(3)
        mid=controller.optional_id("body",f"{camera}_mount")
        lmid=mujoco.mj_name2id(lmodel,mujoco.mjtObj.mjOBJ_BODY,f"{camera}_mount")
        if mid>=0 and lmid>=0:
            mujoco.mj_forward(self.model,self.data)
            offset=self.data.xpos[mid]-ldata.xpos[lmid]
        # 本地模型自带 station_camera；没有就用安装座构造俯视自由相机。
        lcam=mujoco.mj_name2id(lmodel,mujoco.mjtObj.mjOBJ_CAMERA,camera)
        cam_desc=f"model:'{camera}'"
        if lcam<0:
            pose=tuple(ldata.xpos[lmid]) if lmid>=0 else DEFAULT_CAMERA_POSE
            lcam=top_down_camera(pose)
            cam_desc="free top-down"
        viewer=MirrorCameraViewer(
            self.data,lmodel,ldata,lcam,offset,pairs,
            f"{camera} (mesh mirror)",tmp_path=tmp,
        )
        print(
            f"[CAMERA VIEW] mesh mirror: nmesh={lmodel.nmesh}, joints={len(pairs)}/{self.model.njnt}, "
            f"camera={cam_desc}, offset={np.round(offset,3).tolist()}",
            flush=True,
        )
        return viewer

    def close(self):
        if getattr(self,"_viewer",None) is not None:
            self._viewer.close()
            self._viewer=None
        if self.channel is not None:
            self.loop.run_until_complete(self.channel.close())
        self.loop.close()
        asyncio.set_event_loop(None)

def _offline_combined(scene, arm):
    """Generate a temporary XML that include()s the scene and the arm.

    The scene and the arm are separate assets, but the local MuJoCo backend
    needs a single model. MuJoCo resolves meshdir relative to the main XML
    directory, so the temporary file has to sit next to the two source files.
    """
    path = scene.with_name("._offline_combined.xml")
    path.write_text(
        "<?xml version='1.0' encoding='utf-8'?>\n"
        '<mujoco model="ev_battery_offline_combined">\n'
        f'  <include file="{scene.name}"/>\n'
        f'  <include file="{arm.name}"/>\n'
        "</mujoco>\n",
        encoding="utf-8",
    )
    return path

def main():
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--addr",default="localhost:50051")
    p.add_argument("--prefix",default="",help="Exact imported name prefix for the scene, such as scene_two_pack_usda_.")
    p.add_argument("--arm-prefix",default=None,help="Exact imported name prefix for the robot arm when it is a separate asset (e.g. ur5e_arm_usda_). Defaults to --prefix.")
    p.add_argument("--offline",action="store_true",help="Test installed SDK locally, without OrcaStudio.")
    p.add_argument("--xml",type=Path,default=here/"scene_two_pack.xml",help="Used only with --offline.")
    p.add_argument("--arm-xml",type=Path,default=here/"ur5e_arm.xml",help="Robot arm asset merged into the scene for --offline runs.")
    p.add_argument("--list-names",action="store_true",help="Print downloaded joint/site/body names and exit.")
    p.add_argument("--camera-view",nargs="?",const=DEFAULT_CAMERA,default=None,metavar="NAME",help="Open a real-time popup window rendering a MuJoCo camera (default name: station_camera).")
    p.add_argument("--camera-source",choices=("auto","mirror","studio"),default="auto",help="Image source for --camera-view: studio grabs the OrcaStudio viewport via GetCameraFramePNG (online), mirror renders the local mesh model, auto = studio online with mirror fallback.")
    p.add_argument("--fast",action="store_true")
    p.add_argument("--through",choices=PARTS)
    p.add_argument("--report",type=Path,default=here/"orca_run_report.json")
    a=p.parse_args()
    s=c=None
    combined=None
    try:
        offline_xml=None
        if a.offline:
            offline_xml=a.xml.resolve()
            arm_xml=a.arm_xml.resolve()
            # 场景与机械臂是两个独立资产；离线运行需要把它们合成一个模型。
            if (arm_xml.exists() and arm_xml!=offline_xml
                    and arm_xml.name not in offline_xml.read_text(encoding="utf-8")):
                combined=_offline_combined(offline_xml,arm_xml)
                offline_xml=combined.resolve()
        s=OrcaSession(a.addr,offline_xml)
        if a.list_names:
            for obj,count in [(mujoco.mjtObj.mjOBJ_JOINT,s.model.njnt),
                              (mujoco.mjtObj.mjOBJ_ACTUATOR,s.model.nu),
                              (mujoco.mjtObj.mjOBJ_SITE,s.model.nsite),
                              (mujoco.mjtObj.mjOBJ_BODY,s.model.nbody),
                              (mujoco.mjtObj.mjOBJ_GEOM,s.model.ngeom),
                              (mujoco.mjtObj.mjOBJ_CAMERA,s.model.ncam),
                              (mujoco.mjtObj.mjOBJ_EQUALITY,s.model.neq)]:
                print(obj.name,[mujoco.mj_id2name(s.model,obj,i) for i in range(count)])
            return 0
        cfg=Config(realtime=not(a.offline or a.fast))
        c=Controller(s.model,s.data,config=cfg,prefix=a.prefix,arm_prefix=a.arm_prefix,on_frame=s.sync)
        if a.camera_view:
            s.enable_camera_view(c,a.camera_view,a.camera_source)
        c.report["backend"]="orca_sdk_offline" if a.offline else "orca_studio_connected"
        c.report["orca_address"]=None if a.offline else a.addr
        # Imported name prefixes are resolved by name, never by fixed array indices.
        for n in PARTS:
            c.id("equality","fixture_"+n)
        # Orca can omit keyframes during import. Start the demonstration at its ready pose.
        q=c.solve(c.d.site_xpos[c.part_sites["battery_cover"]]+[0,0,.078],
                  c.d.qpos[c.qa].copy())
        c.d.qpos[c.qa]=q;c.d.ctrl[c.aids]=q;c.d.qvel[c.va]=0
        mujoco.mj_forward(c.m,c.d)
        s.sync(c,force=True)
        c.run(a.through)
        s.sync(c,force=True)
        print(f"SUCCESS: {len(c.completed)} parts; backend={c.report['backend']}")
        return 0
    except KeyboardInterrupt:
        print("Stopped by user; held part is not intentionally released.",file=sys.stderr)
        return 130
    except Exception as e:
        print(f"STOPPED: {type(e).__name__}: {e}",file=sys.stderr)
        print("Live mode: import scene_two_pack.xml with assets/, press Run in Orca, verify --addr, --prefix and --arm-prefix.",file=sys.stderr)
        if c is not None:
            c.report["success"]=False
            c.report["error"]=f"{type(e).__name__}: {e}"
        return 1
    finally:
        if c is not None:
            c.save_report(a.report)
        if s is not None:
            s.close()
        if combined is not None:
            combined.unlink(missing_ok=True)

if __name__=="__main__":
    raise SystemExit(main())
