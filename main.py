import numpy as np
from jax import random

from ban.bandits import LSBandit
from ban.agents import LinearThomsonAgent
from ban.trainer import Trainer
from ban.args import Args, get_results_fname


if __name__ == '__main__':
    parser = Args()
    args = parser.parse_args()
    print(f"Begin training with args: {args}")

    args.results_fname = get_results_fname(args)

    rand_key = random.PRNGKey(args.seed)

    bandit = LSBandit(args.action_dim, rand_key, noise_scale=args.reward_noise)
    agent = LinearThomsonAgent(args.action_dim, args.lamb, rand_key)

    trainer = Trainer(bandit, agent, args)

    trainer.train()

    info = trainer.get_info()
    results_path = args.results_dir / args.results_fname

    np.save(results_path, info)
    print(f"Finished training, results saved to {results_path}")

