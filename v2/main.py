import pettingzoo.mpe.simple_v2 as simple
import argument
import maddpg

if __name__ == '__main__':
    args = argument.get_args()

    if args.test_mode:
        max_cycles = args.test_step * (args.num_agent + args.num_adversary)
        env = simple.env(max_cycles = max_cycles, continuous_actions = True).env.env
        policy = maddpg.policy(env, args)
        policy.test()
    else:
        max_cycles = args.num_step * (args.num_agent + args.num_adversary)
        env = simple.env(max_cycles = max_cycles, continuous_actions = True).env.env
        policy = maddpg.policy(env, args)
        policy.train()
