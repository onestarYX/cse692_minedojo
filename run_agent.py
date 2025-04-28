from utils import *
from planner import Planner
from percipient import Percipient
from minedojo.sim import InventoryItem
from pathlib import Path
import minedojo

if __name__ == "__main__":

    GOOGLE_API_KEY = "AIzaSyChmF0glFkgHwSatkXfFQIBEbaW8-PQCUE"
    gemini_model_name = "gemini-2.0-flash"
    taskname = "easy-2-gpt-parsed"
    task = "tasks/"+taskname+".json"
    

    width = 512
    height = 384
    env = minedojo.make(
        task_id="open-ended",
        generate_world_type='flat',
        image_size=(height, width), 
        start_position=dict(x=0,y=4,z=-10,yaw=0,pitch=0)
    )
    obs = env.reset()
    
    planner = Planner(GOOGLE_API_KEY=GOOGLE_API_KEY, gemini_model_name=gemini_model_name)
    percipient = Percipient(GOOGLE_API_KEY=GOOGLE_API_KEY, gemini_model_name=gemini_model_name)
    
    with open(task,'r') as f:
        frames = []
        sub_objective_list = json.load(f)
        for sub_objective in sub_objective_list:
            log_info(f"sub_objective: {sub_objective['description']}")
            
            every_sub_objective_max_retries = 2
            check_result = {}
            
            while every_sub_objective_max_retries >= 0:
    
                # Plan the actions for the current sub-objective
                plan = planner.get_plan(sub_objective=sub_objective, check_result=check_result)
                    
                actions = plan["actions"]
                
                # Execute the actions in Minecraft
                first_entry = actions[0]
                last_entry = actions[-1]
                actions = [first_entry] + actions + [last_entry]
                for entry in actions:
                    args = entry['args']
                    target_pos = np.array((args['x'], args['y']+4, args['z']), dtype=np.float32)
                    block_type = args['block']

                    offset = 8
                    env.teleport_agent(-offset, 4 + offset * 2, -offset, -45, 45)
                    # env.teleport_agent(2, 4 + 4, -8, 0, 20)
                    place_block(env, block_type, target_pos)

                    frames.append(render_rgb(env))
                
                # Check the sub-objective result    
                check_result = percipient.check_sub_objective_success(sub_objective=sub_objective, last_frame = frames[-1])
                
                if check_result["success"] == "true":
                    # Success: Move to the next sub-objective
                    break
                else: 
                    # Failure: Destroy the structure built for the current sub-objective
                    '''
                    for entry in reversed(actions):
                        args = entry['args']
                        target_pos = np.array((args['x'], args['y']+4, args['z']), dtype=np.float32)
                        block_type = 'air'

                        offset = 8
                        env.teleport_agent(-offset, 4 + offset * 2, -offset, -45, 45)
                        # env.teleport_agent(2, 4 + 4, -8, 0, 20)
                        place_block(env, block_type, target_pos)

                        frames.append(render_rgb(env))
                    '''
                    
                    every_sub_objective_max_retries -= 1
                    continue


        # Stop at last frame for 2 seconds
        fps = 15
        last_frame = frames[-1]
        frames = frames + [last_frame] * fps * 2
    
        out_video_file = f"output/"+taskname+".avi"
        out = cv2.VideoWriter(out_video_file, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

        for frame in frames:
            out.write(frame)  # Each frame must be (H, W, 3) and dtype=uint8

        out.release()
        
        cv2.imwrite(f"output/"+taskname+".png",last_frame)
    
    # while(True):
    #     pass
    env.close()
