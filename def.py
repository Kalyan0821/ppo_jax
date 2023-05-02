import gymnax

for env_name in ["Acrobot-v1", 
                 "Asterix-MinAtar", 
                 "Breakout-MinAtar", 
                 "CartPole-v1", 
                 "Freeway-MinAtar", 
                 "MountainCar-v0", 
                 "SpaceInvaders-MinAtar"]:
    env, env_params = gymnax.make(env_name)
    print(env_params.max_steps_in_episode, '\n')
