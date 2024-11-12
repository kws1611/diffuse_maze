import json
import numpy as np
from os.path import join
import torch

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils


class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer
goal_state = torch.tensor([7.0, 9.0], dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
start_state = torch.tensor([2.0, 2.0], dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
policy = Policy(diffusion, dataset.normalizer, start_state, goal_state,
                    lambda_energy=0.1, lambda_distance=0.1, lambda_obstacle=1.0,
                    obstacle_padding=2)

# Set start and goal state in the environment if applicable
if hasattr(env, 'set_start_state'):
    env.set_start_state(start_state.cpu().numpy())
if hasattr(env, 'set_goal_state'):
    env.set_goal_state(goal_state.cpu().numpy())

#---------------------------------- main loop ----------------------------------#

observation = env.reset()
if args.conditional:
    print('Resetting target')
    env.set_target()

## set conditioning xy position to be the goal
target = goal_state.cpu().numpy()
cond = {
    diffusion.horizon - 1: torch.tensor([*target, 0, 0], dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu'),
}

## observations for rendering
rollout = [observation.copy()]

total_reward = 0
for t in range(env.max_episode_steps):

    state = torch.tensor(env.state_vector(), dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')  # Convert state to tensor and move to appropriate device

    ## can replan if desired, but the open-loop plans are good enough for maze2d
    ## that we really only need to plan once
    if t == 0:
        cond[0] = torch.tensor(observation, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')

        samples = policy.model.p_sample_loop((args.batch_size, policy.model.horizon, policy.model.transition_dim), cond)
        actions = samples[:, :, :policy.model.action_dim]  # Extract actions from samples
        sequence = samples[:, :, policy.model.action_dim:]  # Extract observations from samples
        action = actions[0, 0]  # Use the first action of the best trajectory

    if t < len(sequence) - 1:
        next_waypoint = sequence[0, t + 1]
    else:
        next_waypoint = sequence[0, -1].clone()
        next_waypoint[2:] = 0

    ## can use actions or define a simple controller based on state predictions
    action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
    action = action.cpu().numpy()  # Convert action back to NumPy array for environment

    next_observation, _, terminal, _ = env.step(action)  # Ignore the environment-calculated reward

    # Use custom reward from the policy instead
    reward = policy.new_reward_function(torch.tensor(next_observation, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu').unsqueeze(0), obstacles=torch.tensor([[3.0, 3.0], [5.0, 5.0]], dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')).item()  # Example custom reward calculation
    total_reward += reward

    score = env.get_normalized_score(total_reward)
    print(
        f't: {t} | r: {reward:.2f} | R: {total_reward:.2f} | score: {score:.4f} | {action}'
    )

    if 'maze2d' in args.dataset:
        xy = next_observation[:2]
        goal = env.unwrapped._target
        print(
            f'maze | pos: {xy} | goal: {goal}'
        )

    ## update rollout observations
    rollout.append(next_observation.copy())

    if t % args.vis_freq == 0 or terminal:
        fullpath = join(args.savepath, f'{t}.png')

        if t == 0:
            renderer.composite(fullpath, sequence.cpu().numpy(), ncol=1)

        renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout, dtype=np.float32)[None], ncol=1)

    if terminal:
        break

    observation = next_observation

# save result as a json file
json_path = join(args.savepath, 'rollout.json')
json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
    'epoch_diffusion': diffusion_experiment.epoch}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)

