import minedojo
import numpy as np
from agent_utils import *
import json
import cv2
from pathlib import Path


if __name__ == '__main__':
    input_file = Path("../data/actions_v2/easy-5-gpt-parsed-gemini-planned-zero-shot.json")
    # data will be a list of dictionariesz
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    width=512
    height=384
    env = minedojo.make(
        task_id="open-ended",
        generate_world_type='flat',
        image_size=(height, width),
        start_position=dict(x=0, y=4, z=-10, yaw=0, pitch=0)
    )
    obs = env.reset()

    frames = []
    first_entry = data[0]
    last_entry = data[-1]
    data = [first_entry] + data + [last_entry]
    for entry in data:
        args = entry['args']
        target_pos = np.array((args['x'], args['y'], args['z']), dtype=np.float32)
        block_type = args['block']

        offset = 8
        env.teleport_agent(-offset, 4 + offset * 2, -offset, -45, 45)
        # env.teleport_agent(2, 4 + 4, -8, 0, 20)
        place_block(env, block_type, target_pos)

        frames.append(render_rgb(env))

    # Stop at last frame for 2 seconds
    fps = 15
    last_frame = frames[-1]
    frames = frames + [last_frame] * fps * 2
    
    out_video_file = input_file.parent / f"{input_file.stem}.mp4"
    out = cv2.VideoWriter(out_video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame in frames:
        out.write(frame)  # Each frame must be (H, W, 3) and dtype=uint8

    out.release()
    
    # while(True):
    #     pass
    env.close()