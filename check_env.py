from __future__ import annotations

import mujoco

from env import FactoryFloorEnv

env = FactoryFloorEnv()
try:
    observation, _ = env.reset()
    mujoco.mj_forward(env.model, env.data)

    print({"nq": env.model.nq, "nu": env.model.nu, "time": env.data.time})
    print(
        {
            "observation": {
                name: {"shape": value.shape, "dtype": str(value.dtype)}
                for name, value in observation.items()
            }
        }
    )
    print(env.describe_task())
finally:
    env.close()
