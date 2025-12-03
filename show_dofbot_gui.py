import pybullet as p
import time
import os

p.connect(p.GUI)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

urdf_path = os.path.join(
    BASE_DIR,
    "dofbot",
    "assets",
    "dofbot",
    "dofbot.urdf"
)

print("Using URDF path:")
print(urdf_path)

if not os.path.exists(urdf_path):
    raise FileNotFoundError(f"❌ URDF not found:\n{urdf_path}")

robot_id = p.loadURDF(
    urdf_path,
    basePosition=[0, 0, 0],
    useFixedBase=True
)

print("✅ DOFBOT loaded, robot_id =", robot_id)

while p.isConnected():
    p.stepSimulation()
    time.sleep(1 / 240)
