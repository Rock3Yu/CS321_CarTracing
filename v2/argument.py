import argparse

def get_args():
    parser = argparse.ArgumentParser()

    #environment
    parser.add_argument('--num_agent', type=int, default=3, help='number of agents')
    parser.add_argument('--num_adversary', type=int, default=0, help='number of adversaries')

    #training
    parser.add_argument('--num_epoch', type=int, default=int(1e3), help='number of epoches')
    parser.add_argument('--num_step', type=int, default=250, help='number of steps for a single epoch')
    parser.add_argument('--lr_a', type=float, default=1e-4, help='learning rate of actor')
    parser.add_argument('--lr_c', type=float, default=1e-3, help='learning rate of critic')
    parser.add_argument('--gamma', type=float, default=.95, help='discount factor for future')
    parser.add_argument('--tau', type=float, default=.01, help='soft update factor')
    parser.add_argument('--random_step', type=int, default=int(5e3), help='# of steps of random action')
    parser.add_argument('--update_freq', type=int, default=50, help='# of steps between each update')
    parser.add_argument('--memory_size', type=int, default=int(1e6), help='size of memory buffer')
    parser.add_argument('--batch_size', type=int, default=512, help='size of training batch')
    parser.add_argument('--load', type=bool, default=False, help='whether to load model')

    #testing
    parser.add_argument('--test_mode', type=bool, default=False, help='whether to perform test')
    parser.add_argument('--test_epoch', type=int, default=20, help='number of epoches of test')
    parser.add_argument('--test_step', type=int, default=50, help='number of steps for a single epoch of test')

    #others
    parser.add_argument('--render', type=bool, default=True, help='whether to render')
    parser.add_argument('--model_dir', type=str, default='./model', help='directory of model')
    parser.add_argument('--log_dir', type=str, default='./log', help='directory of log')

    args = parser.parse_args()
    return args
