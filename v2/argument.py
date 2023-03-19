import argparse

def get_args():
    parser = argparse.ArgumentParser()

    #environment
    parser.add_argument('--env_name', type=str, default='spread', help='name of environment, [simple/tag/spread]')
    parser.add_argument('--num_agent', type=int, default=3, help='number of agents')
    parser.add_argument('--num_adversary', type=int, default=3, help='number of adversaries')

    #training
    parser.add_argument('--num_epoch', type=int, default=int(1e4), help='# of epoches')
    parser.add_argument('--num_step', type=int, default=25, help='# of steps for a single epoch')
    parser.add_argument('--lr_a', type=float, default=1e-4, help='learning rate of actor')
    parser.add_argument('--lr_c', type=float, default=1e-3, help='learning rate of critic')
    parser.add_argument('--gamma', type=float, default=.95, help='discount factor for future')
    parser.add_argument('--tau', type=float, default=.01, help='soft update factor')
    parser.add_argument('--noise', type=float, default=.3, help='rate of noise')
    parser.add_argument('--decay', type=float, default=.99, help='rate of noise decay')
    parser.add_argument('--random_step', type=int, default=int(2e3), help='# of steps of random action')
    parser.add_argument('--update_freq', type=int, default=1, help='# of steps between each update')
    parser.add_argument('--memory_size', type=int, default=int(1e6), help='size of memory buffer')
    parser.add_argument('--batch_size', type=int, default=64, help='size of training batch')
    parser.add_argument('--load', type=bool, default=False, help='whether to load model')
    parser.add_argument('--test_freq', type=int, default=100, help='# of epoches between each test')
    parser.add_argument('--test_epoch', type=int, default=10, help='# of epoches of test')
    parser.add_argument('--save_freq', type=int, default=1000, help='# of epoches between each model save')
    parser.add_argument('--load_epoch', type=int, default=10000, help='load model name')
    parser.add_argument('--local_ratio', type=float, default=.7, help='ratio of local reward')

    #testing
    parser.add_argument('--test_mode', type=bool, default=True, help='whether to perform test')

    #render
    parser.add_argument('--render_mode', type=str, default=None, help='render mode, [human/rgb_array]')  #if not git then not output picture
    parser.add_argument('--fps', type=int, default=10, help='fps for render')
    parser.add_argument('--render_freq', type=int, default=1000, help='# of epoches between each render')
    #for unchange local ratio is l7
    #path
    PATH_NAME = 'spread_l7'
    parser.add_argument('--overwrite', type=bool, default=False, help='file protection')
    parser.add_argument('--model_dir', type=str, default=f'./model/{PATH_NAME}', help='directory of model')
    parser.add_argument('--log_dir', type=str, default=f'./log/{PATH_NAME}', help='directory of log')
    parser.add_argument('--image_dir', type=str, default=f'./image/{PATH_NAME}', help='directory of render image')
    parser.add_argument('--tensorboard_dir', type=str, default=f'./log/tensorboard/{PATH_NAME}', help='directory of tensorboard log')

    args = parser.parse_args()
    return args
