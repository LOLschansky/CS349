import numpy as np
import src.random


class MultiArmedBandit:
    """
    MultiArmedBandit reinforcement learning agent.

    Arguments:
      epsilon - (float) The probability of randomly exploring the action space
        rather than exploiting the best action.
    """

    def __init__(self, epsilon=0.2):
        self.epsilon = epsilon

    def fit(self, env, steps=1000, num_bins=100):
        """
        Trains the MultiArmedBandit on an OpenAI Gymnasium environment.

        See page 32 of Sutton and Barto's book Reinformcement Learning for
        pseudocode (http://incompleteideas.net/book/RLbook2020.pdf).
        Initialize your parameters as all zeros. For the step size, use
        1/N, where N is the number of times the current action has been
        performed. (This is the version of Bandits we saw in lecture before
        we introduced alpha). Use an epsilon-greedy approach to pick actions.

        See (https://gymnasium.farama.org/) for examples of how to use the OpenAI
        Gymnasium Environment interface.

        In every step of the fit() function, you should sample
            two random numbers using functions from `src.random`.
            1.  First, use either `src.random.rand()` or `src.random.uniform()`
                to decide whether to explore or exploit.
            2. Then, use `src.random.choice` or `src.random.randint` to decide
                which action to choose. Even when exploiting, you should make a
                call to `src.random` to break (possible) ties.

        Please don't use `np.random` functions; use the ones from `src.random`!
        Please do not use `env.action_space.sample()`!

        Hints:
          - Use env.action_space.n and env.observation_space.n to get the
            number of available actions and states, respectively.
          - Remember to reset your environment at the end of each episode. To
            do this, call env.reset() whenever the value of "terminated or truncated" returned
            from env.step() is True.
          - When choosing to exploit the best action rather than exploring,
            do not use np.argmax: it will deterministically break ties by
            choosing the lowest index of among the tied values. Instead,
            please *randomly choose* one of those tied-for-the-largest values.
          - MultiArmedBandit treats all environment states the same. However,
            in order to have the same API as agents that model state, you must
            explicitly return the state-action-values Q(s, a). To do so, just
            copy the action values learned by MultiArmedBandit S times, where
            S is the number of states.

        Arguments:
          env - (Env) An OpenAI Gymnasium environment with discrete actions and
            observations. See the OpenAI Gymnasium documentation for example use
            cases (https://gymnasium.farama.org/api/env/).
          steps - (int) The number of actions to perform within the environment
            during training.

        Returns:
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.
          rewards - (np.array) A 1D sequence of averaged rewards of length `num_bins`.
            Let s = int(np.ceil(steps / `num_bins`)), then rewards[0] should
            contain the average reward over the first s steps, rewards[1]
            should contain the average reward over the next s steps, etc.
            Please note that: The total number of steps will not always divide evenly by the 
            number of bins. This means the last group of steps may be smaller than the rest of 
            the groups. In this case, we can't divide by s to find the average reward per step 
            because we have less than s steps remaining for the last group.
        """

        # set up Q function, rewards
        n_actions, n_states = env.action_space.n, env.observation_space.n
        self.Q = np.zeros(n_actions)
        self.N = np.zeros(n_actions)
        avg_rewards = np.zeros([num_bins])
        all_rewards = []

        # reset environment before your first action
        env.reset()

        for step in range(steps):
          # draw random number
          rand_num_expl = src.random.uniform(0.0, 1.0)

          # with a probability of epsilon, explore, otherwise, exploit
          if rand_num_expl < self.epsilon:
            action = src.random.randint(0, n_actions)
          else:
            # find the maximum value in the vector
            max_value = np.max(self.Q)
            
            # find the indices of all elements equal to the maximum value, s
            indices = np.where(self.Q == max_value)[0]
            rand_num_choice = src.random.randint(0, len(indices))
            action = indices[rand_num_choice]

          # take a step
          step_dat = env.step(action)

          # update N
          self.N[action] += 1

          # reset environment if truncated or terminated
          truncated = step_dat[3]
          terminated = step_dat[2]
          if truncated or terminated:
            env.reset()

          # update all_rewards
          reward = step_dat[1]
          all_rewards.append(reward)

          # update Q
          self.Q[action] += 1/self.N[action]*(reward - self.Q[action])

        # compute average rewards for bins
        bin_size = int(np.ceil(steps / num_bins))
        for bin_ind in range(num_bins):
          start_ind = bin_ind * bin_size
          end_ind = start_ind + bin_size if bin_ind < num_bins - 1 else len(all_rewards)
          avg_rewards[bin_ind] = np.mean(all_rewards[start_ind:end_ind]) if end_ind > start_ind else 0

        # create state-action values
        state_action_values = np.tile(self.Q, (n_states, 1))

        # return values
        return state_action_values, avg_rewards


    def predict(self, env, state_action_values):
        """
        Runs prediction on an OpenAI environment using the policy defined by
        the MultiArmedBandit algorithm and the state action values. Predictions
        are run for exactly one episode. Note that one episode may produce a
        variable number of steps.

        Hints:
          - You should not update the state_action_values during prediction.
          - Exploration is only used in training. During prediction, you
            should only "exploit."
          - You should use a loop to predict over each step in an episode until
            it terminates by returning `terminated or truncated=True`.
          - When choosing to exploit the best action, do not use np.argmax: it
            will deterministically break ties by choosing the lowest index of
            among the tied values. Instead, please *randomly choose* one of
            those tied-for-the-largest values.

        Arguments:
          env - (Env) An OpenAI Gymnasium environment with discrete actions and
            observations. See the OpenAI Gymnasium documentation for example use
            cases (https://gymnasium.farama.org/api/env/).
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.

        Returns:
          states - (np.array) The sequence of states visited by the agent over
            the course of the episode. Does not include the starting state.
            Should be of length K, where K is the number of steps taken within
            the episode.
          actions - (np.array) The sequence of actions taken by the agent over
            the course of the episode. Should be of length K, where K is the
            number of steps taken within the episode.
          rewards - (np.array) The sequence of rewards received by the agent
            over the course  of the episode. Should be of length K, where K is
            the number of steps taken within the episode.
        """

        # reset environment before your first action
        reset_dat = env.reset()

        state = reset_dat[0]

        states, actions, rewards = [], [], []

        terminated, truncated = False, False

        while not (terminated or truncated):

          action_values = state_action_values[state]

          # find the maximum value in the vector
          max_value = np.max(action_values)
            
          # find the indices of all elements equal to the maximum value
          indices = np.where(action_values == max_value)[0]
          rand_num_choice = src.random.randint(0, len(indices))
          action = indices[rand_num_choice]

          # step 
          next_state, reward, terminated, truncated, _ = env.step(action)

          # add data to corresponding vectors
          states.append(next_state)
          actions.append(action)
          rewards.append(reward)

          # update the current state
          state = next_state

        return np.array(states), np.array(actions), np.array(rewards)

