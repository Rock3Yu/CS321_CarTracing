import pettingzoo.mpe.simple_v2 as simple
import pettingzoo.mpe.simple_tag_v2 as tag
import argument
import maddpg
import os

if __name__ == '__main__':
    args = argument.get_args()
    if not os.path.exists(args.image_dir):
        os.mkdir(args.image_dir)

    if args.test_mode:
        max_cycles = args.test_step * (args.num_agent + args.num_adversary)
        # env = simple.env(max_cycles = max_cycles, continuous_actions = True, render_mode = args.render_mode).env.env
        env = tag.env(num_good = args.num_adversary, num_adversaries = args.num_agent, max_cycles = max_cycles, continuous_actions = True, render_mode = args.render_mode).env.env
        policy = maddpg.policy(env, args)
        policy.test()
    else:
        max_cycles = args.num_step * (args.num_agent + args.num_adversary)
        # env = simple.env(max_cycles = max_cycles, continuous_actions = True, render_mode = args.render_mode).env.env
        env = tag.env(num_good = args.num_adversary, num_adversaries = args.num_agent, max_cycles = max_cycles, continuous_actions = True, render_mode = args.render_mode).env.env
        policy = maddpg.policy(env, args)
        policy.train()
