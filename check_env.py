from __future__ import annotations

import mujoco


model = mujoco.MjModel.from_xml_path("third_party/menagerie/universal_robots_ur5e/scene.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print({"nq": model.nq, "nu": model.nu, "time": data.time})
