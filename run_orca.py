#!/usr/bin/env python3
"""OrcaStudio/OrcaLab bridge using the installed OrcaGymLocal backend.

Physics runs in the SDK's local MuJoCo model; gym.render() synchronizes qpos
(including all seven freejoint parts) to the running Orca scene.
"""
import argparse
import asyncio
from pathlib import Path
import sys
import mujoco
from disassembly import Controller, Config, MotionError, PARTS

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
        async def connect():
            if self.offline:
                self.gym=OrcaGymLocal(None,skip_grpc_load=True,local_xml_path=str(offline_xml))
                await self.gym.init_simulation(str(offline_xml))
            else:
                self.channel=grpc.aio.insecure_channel(addr,options=[
                    ("grpc.max_receive_message_length",1024**3),
                    ("grpc.max_send_message_length",1024**3)])
                await asyncio.wait_for(self.channel.channel_ready(),timeout)
                self.gym=OrcaGymLocal(GrpcServiceStub(self.channel))
                xml=await asyncio.wait_for(self.gym.load_model_xml(),120)
                await asyncio.wait_for(self.gym.pause_simulation(),timeout)
                await self.gym.init_simulation(xml)
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
        # Controller frames are 20 ms (50 Hz). Send every new frame; the old
        # 1/30 threshold actually produced uneven 40 ms updates (25 Hz).
        if not force and controller.d.time <= self.last_frame + 1e-12:
            return
        self.gym.update_data()
        if not self.offline:
            self.loop.run_until_complete(asyncio.wait_for(self.gym.render(),self.timeout))
        self.last_frame=float(controller.d.time)

    def close(self):
        if self.channel is not None:
            self.loop.run_until_complete(self.channel.close())
        self.loop.close()
        asyncio.set_event_loop(None)

def main():
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--addr",default="localhost:50051")
    p.add_argument("--prefix",default="",help="Exact imported name prefix, such as Robot1_.")
    p.add_argument("--offline",action="store_true",help="Test installed SDK locally, without OrcaStudio.")
    p.add_argument("--xml",type=Path,default=here/"scene_optimized.xml",help="Used only with --offline.")
    p.add_argument("--list-names",action="store_true",help="Print downloaded joint/site/body names and exit.")
    p.add_argument("--fast",action="store_true")
    p.add_argument("--through",choices=PARTS)
    p.add_argument("--report",type=Path,default=here/"orca_run_report.json")
    a=p.parse_args()
    s=c=None
    try:
        s=OrcaSession(a.addr,a.xml.resolve() if a.offline else None)
        if a.list_names:
            for obj,count in [(mujoco.mjtObj.mjOBJ_JOINT,s.model.njnt),
                              (mujoco.mjtObj.mjOBJ_ACTUATOR,s.model.nu),
                              (mujoco.mjtObj.mjOBJ_SITE,s.model.nsite),
                              (mujoco.mjtObj.mjOBJ_BODY,s.model.nbody),
                              (mujoco.mjtObj.mjOBJ_GEOM,s.model.ngeom),
                              (mujoco.mjtObj.mjOBJ_EQUALITY,s.model.neq)]:
                print(obj.name,[mujoco.mj_id2name(s.model,obj,i) for i in range(count)])
            return 0
        cfg=Config(realtime=not(a.offline or a.fast))
        c=Controller(s.model,s.data,config=cfg,prefix=a.prefix,on_frame=s.sync)
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
        print("Live mode: import scene_optimized.xml with assets/, press Run in Orca, verify --addr and --prefix.",file=sys.stderr)
        if c is not None:
            c.report["success"]=False
            c.report["error"]=f"{type(e).__name__}: {e}"
        return 1
    finally:
        if c is not None:
            c.save_report(a.report)
        if s is not None:
            s.close()

if __name__=="__main__":
    raise SystemExit(main())
