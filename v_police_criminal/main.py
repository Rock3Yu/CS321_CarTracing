import argument
import maddpg
import maddpg_mpe
import os

import Car_Tracing_Scenario as scenario
import pettingzoo.mpe.simple_spread_v2 as spread
import pettingzoo.mpe.simple_tag_v2 as tag

if __name__ == '__main__':
    args = argument.get_args()
    if not args.test_mode and not args.overwrite and os.path.exists(args.model_dir):
        raise FileExistsError('path already exist, if you want to overwrite, set the argument to True')

    max_cycles = args.num_step * args.num_agent
    # env = scenario.env(
    #     max_cycles = max_cycles, 
    #     map_path = args.map_path,
    #     render_mode = args.render_mode, 
    #     local_ratio = args.local_ratio
    #     ).env.env
    env = spread.env(N=args.num_agent,continuous_actions = True, render_mode = 'rgb_array', local_ratio = args.local_ratio).env.env
    # env = tag.env(continuous_actions = True, render_mode = 'rgb_array').env.env
    policy = maddpg_mpe.policy(env, args)
    policy.run()
