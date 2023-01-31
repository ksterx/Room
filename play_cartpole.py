import argparse
import shutil
from tkinter.filedialog import askopenfilename

import gym

from room.agents import DQN
from room.envs import register_env

INITIALDIR = "/Users/ksterx/Library/CloudStorage/GoogleDrive-ishikawa-kosuke259@g.ecc.u-tokyo.ac.jp/My Drive/Development/Python Projects/Room/experiments/results"

parser = argparse.ArgumentParser()
parser.add_argument("--weight_path", type=str, default=None)
parser.add_argument("-r", "--use_recent", action="store_true")
args = parser.parse_args()

env = gym.make("CartPole-v1", render_mode="human")
state_shape = env.observation_space.shape[0]
action_shape = env.action_space.n
env = register_env(env)


# Copy the recent weight file as sample_weight.pt
def copy():
    shutil.copy2(weight_path, "sample_weight.pt")


if args.weight_path is None:
    if args.use_recent:
        weight_path = "sample_weight.pt"
    else:
        weight_path = askopenfilename(initialdir=INITIALDIR)
        copy()
else:
    weight_path = args.weight_path
    copy()


agent = DQN(model="mlp3")
agent.initialize(state_shape=state_shape, action_shape=action_shape, training=False)
agent.load(weight_path)
agent.play(env, num_eps=5)
print("Done!")
