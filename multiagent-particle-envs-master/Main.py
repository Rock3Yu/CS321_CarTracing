import make_env

if __name__ == '__main__':
    for i in range(1):
        env = make_env.make_env("simple_speaker_listener")
        env.render()
        # test
