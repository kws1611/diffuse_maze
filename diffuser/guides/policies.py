import torch
import numpy as np

class Policy:
    def __init__(self, model, normalizer, start_state, goal_state, lambda_energy=0.1, lambda_distance=1.0, lambda_obstacle=1.0, obstacle_padding=0.0):
        """
        Initialize the Policy class with a diffusion model, normalizer, start and goal state, energy scaling factor, distance scaling factor, obstacle scaling factor, and obstacle padding.
        
        Parameters:
        model (object): Trained diffusion model used for sampling trajectories.
        normalizer (object): Data normalizer for states and actions.
        start_state (tensor): The starting state for the trajectory.
        goal_state (tensor): The goal state that the policy is attempting to reach.
        lambda_energy (float): Scaling factor for the energy penalty in the reward function.
        lambda_distance (float): Scaling factor for the minimum distance reward.
        lambda_obstacle (float): Scaling factor for the obstacle avoidance penalty.
        obstacle_padding (float): Padding around obstacles to adjust avoidance behavior.
        """
        self.model = model  # Diffusion model for sampling
        self.normalizer = normalizer  # Normalizer for state and action
        self.start_state = start_state  # Start state for planning
        self.goal_state = goal_state  # Goal state for planning
        self.lambda_energy = lambda_energy  # Scaling factor for energy penalty
        self.lambda_distance = lambda_distance  # Scaling factor for minimum distance reward
        self.lambda_obstacle = lambda_obstacle  # Scaling factor for obstacle avoidance
        self.obstacle_padding = obstacle_padding  # Padding around obstacles

    def new_reward_function(self, trajectory, obstacles):
        """
        Custom reward function that encourages the shortest path possible by rewarding shorter distances to the goal and avoiding obstacles.
        
        Parameters:
        trajectory (tensor): The sequence of states and actions to evaluate.
        obstacles (tensor): A tensor containing the positions of obstacles.
        
        Returns:
        tensor: The computed reward for the given trajectory.
        """
        # Calculate the distance to the goal state
        goal_distance = torch.norm(trajectory[-1, :2] - self.goal_state)  # Negative reward for distance to goal

        # Assuming the last two columns in the trajectory are actions, compute energy penalty
        actions = trajectory[:, -2:]  # Extract actions from the trajectory
        energy_penalty = torch.sum(torch.square(actions))  # Penalize the sum of squared actions for energy use
        
        # Obstacle avoidance penalty with padding
        obstacle_penalty = 0
        for obstacle in obstacles:
            distances_to_obstacle = torch.norm(trajectory[:, :2] - obstacle, dim=1) - self.obstacle_padding
            obstacle_penalty += torch.sum(torch.exp(-distances_to_obstacle))  # Penalize proximity to obstacles
        
        # Return the total reward as a combination of goal distance reward, energy penalty, and obstacle avoidance penalty
        return -goal_distance - self.lambda_energy * energy_penalty - self.lambda_obstacle * obstacle_penalty

    def __call__(self, cond, obstacles, batch_size=1):
        """
        Generate actions based on the conditions provided using guided sampling with the diffusion model.
        
        Parameters:
        cond (dict): Conditioning information for the model to generate trajectories.
        obstacles (tensor): A tensor containing the positions of obstacles.
        batch_size (int): Number of trajectories to sample in a batch.
        
        Returns:
        action (tensor): The next action to take based on the best trajectory.
        samples (object): The sampled trajectories generated by the diffusion model.
        """
        # Add start state to conditioning
        cond[0] = self.start_state

        # Generate initial sample of trajectories from the diffusion model
        samples = self.model.p_sample_loop((batch_size, self.model.horizon, self.model.transition_dim), cond)
        
        # Extract trajectories from the generated samples
        trajectories = samples[:, :, :self.model.transition_dim]

        # Compute rewards for each of the generated trajectories
        rewards = []
        for trajectory in trajectories:
            reward = self.new_reward_function(trajectory, obstacles)
            rewards.append(reward)
        rewards = torch.tensor(rewards)
        
        # Select the best trajectory based on the computed rewards
        best_trajectory = trajectories[torch.argmax(rewards)]
        action = best_trajectory[0, -2:]  # Extract the first action of the best trajectory
        
        return action, samples

# Example usage
if __name__ == "__main__":
    # Assuming model, normalizer, start_state, goal_state, and obstacles are defined elsewhere
    pretrained_diffusion_model = ...  # Replace with your trained diffusion model
    normalizer = ...  # Replace with your data normalizer
    start_state = torch.tensor([0.0, 0.0])  # Replace with the specific start state
    goal_state = torch.tensor([1.0, 0.5])  # Replace with the specific goal state
    obstacles = torch.tensor([[3.0, 3.0], [5.0, 5.0]])  # Replace with actual obstacle positions
    obstacle_padding = 0.5  # Example padding value
    
    # Instantiate the Policy
    policy = Policy(pretrained_diffusion_model, normalizer, start_state, goal_state, lambda_energy=0.1, lambda_distance=1.0, lambda_obstacle=1.0, obstacle_padding=obstacle_padding)
    
    # Example condition for planning (e.g., initial state)
    cond = {}  # Replace with actual conditioning data
    
    # Generate the next action
    action, samples = policy(cond, obstacles)
    print(f"Selected Action: {action}")

