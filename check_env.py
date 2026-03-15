from __future__ import annotations

import mujoco

from env import FactoryFloorEnv

env = FactoryFloorEnv()
mujoco.mj_forward(env.model, env.data)

print({"nq": env.model.nq, "nu": env.model.nu, "time": env.data.time})
print(env.describe_task())
