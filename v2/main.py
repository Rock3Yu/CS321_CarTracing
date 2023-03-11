import pettingzoo.mpe.simple_v2 as simple
import pettingzoo.mpe.simple_tag_v2 as tag
import pettingzoo.mpe.simple_spread_v2 as spread
import argument
import maddpg
import os

if __name__ == '__main__':
    args = argument.get_args()
    if args.env_name not in ['simple', 'tag','spread']:
        raise NotImplementedError('environment not supported')
    if not args.test_mode and not args.overwrite and (os.path.exists(args.log_dir) or os.path.exists(args.image_dir)):
        raise FileExistsError('path already exist, if you want to overwrite, set the argument to True')
    
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    for dir in ['/actor/', '/critic/']:
        if not os.path.exists(args.model_dir+dir):
            os.mkdir(args.model_dir+dir)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(args.image_dir):
        os.mkdir(args.image_dir)
    if not os.path.exists(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)


    max_cycles = args.num_step * (args.num_agent + args.num_adversary)
    if args.env_name == 'simple':
        env = simple.env(max_cycles=max_cycles , continuous_actions=True,
                        render_mode=args.render_mode).env.env
    if args.env_name == 'tag':
        env = tag.env(num_good = args.num_agent, num_adversaries = args.num_adversary,
                    max_cycles = max_cycles, continuous_actions = True, render_mode = args.render_mode).env.env
    if args.env_name == 'spread':
        env = spread.env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=True).env.env

    policy = maddpg.policy(env, args)
    policy.run()
