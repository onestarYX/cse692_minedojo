from typing import Any
import minedojo
import numpy as np
import cv2

def place_block(env: Any, block_type: str, pos: np.ndarray):
    # Get the agent's position
    null_act = env.action_space.no_op()
    obs, reward, done, info = env.step(null_act)
    agent_pos = obs['location_stats']['pos']

    rel_pos = pos - agent_pos
    env.set_block(block_type, rel_pos)

def render_rgb(env):
    null_act = env.action_space.no_op()
    obs, reward, done, info = env.step(null_act)
    frame = np.transpose(obs['rgb'], (1, 2, 0))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


if __name__ == "__main__":
    env = minedojo.make(
        task_id="open-ended",
        generate_world_type='flat',
        image_size=(384, 512),
        start_position=dict(x=0, y=4, z=-10, yaw=0, pitch=0)
    )

    obs = env.reset()

    for i in range(50):
        env.teleport_agent(0, 20, -10, 0, 45)
        place_block(env, "diamond_ore", np.array([i, 4, 0]))

    env.close()