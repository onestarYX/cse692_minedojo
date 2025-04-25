import minedojo

env = minedojo.make(
    task_id="open-ended",
    generate_world_type='flat',
    image_size=(384, 512),
    start_position=dict(x=0, y=4, z=0, yaw=0, pitch=0)
)
obs = env.reset()
print(obs['location_stats']['pitch'])

null_act = env.action_space.no_op()
# act = env.action_space.no_op()
# act[3] = 6
# obs, reward, done, info = env.step(act)
# print(obs["location_stats"]["pos"])
# print(obs['location_stats']['pitch'])

# x to left, z to up

target_level = 4
for i in range(50):
    # act = env.action_space.no_op()
    # act[0] = 1
    # obs, reward, done, info = env.step(act)

    env.teleport_agent(0, 4, 0, 0, 0)
    # obs, reward, done, info = env.step(null_act)
    # rel_y = target_level - obs['location_stats']['pos'][1]
    env.set_block("diamond_ore", [i, 0, 10])


env.close()