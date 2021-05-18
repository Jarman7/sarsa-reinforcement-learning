# Tom Jarman, SARSA Reinforcement Learning
# This code is an adaptation of COM3240 - Adaptive Intelligence Lab 8
# Found at https://vle.shef.ac.uk/bbcswebdav/pid-5141827-dt-content-rid-34052066_1/xid-34052066_1

#README
# Setup
# There main program is ran through the main() function. Parameters can be set here for:
# * repetitions - The number of experiments
# * trials - The number of trials per experiement
# * max_steps - The maximum number of steps per walk
# * learning_rate
# * epsilon - For epsilon-greedy policy, 0 means just greedy
# * gamma - discount factor
# * OUTPUT_STATES_EXPLORED - Boolean determines if the average number of states explored in the experiments is output

# Optional parameters to pass to homing() function, these default to 0 or False:
# * lambda_val - The lambda value used in SARSA(Lambda) learning
# * SARSA_LAMBDA - Boolean, determines if SARSA(Lambda) learning is used
# * PLOT_DIRECTIONS - Boolean, determines if preferred directions for each state at the end of the experiment

# The main() function is automatically ran.

import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.ndimage import gaussian_filter1d

def homing(trials, max_steps, learning_rate, discount_factor, exploration_factor, lambda_val=0, SARSA_LAMBDA=False, PLOT_DIRECTIONS=False):
    # Carries out SARSA or SARSA(Lambda) reinforcement learning with a single layer NN
    # Outputs are determined by input parameters

    # Initialisation
    Y = 10
    X = 10
    R = 1
    MAX_STEP_PENALTY = -1
    #walls = [[0,5],[1,5],[2,5],[3,5],[9,3],[8,3],[7,3],[6,3],[5,3],[6,7],[6,8],[6,9]] # Array of walls coordinates
    walls = [] # Used when no walls are needed

    # State and action initialisation
    states_explored = []
    states = X * Y
    state_matrix = np.eye(states)
    action_effects = [["N", -1], ["E", +1], ["S", +1], ["W", -1]]
    actions = len(action_effects)

    # Reward location setup
    reward_location = [random.randint(0, X-1), random.randint(0, Y-1)] # Reward location
    #reward_location = [0,0] # Fixed reward for Q5
    #reward_location = [9,2] # Fixed reward for Q7
    
    # NN Setup
    reward_index = np.ravel_multi_index(reward_location, dims=(X, Y), order="F")
    weights = np.random.rand(actions, states)
    
    # Learning Curve
    learning_curve = np.zeros((trials))
    
    # Eligibility trace SARSA(Lambda)
    elig_old = np.zeros((actions, states))

    # Trials
    for trial in range(trials):

        # Initialisation
        start_position = generate_start(reward_location, X, Y)
        # Reduces start position to index
        start_index = np.ravel_multi_index(start_position, dims=(X, Y), order="F")
        # Calculates optimum path to solution assuming no walls, lower bound
        min_steps = abs(start_position[0] - reward_location[0]) + abs(start_position[1] - reward_location[1])
        # Update state to match start
        state = start_position
        state_index = start_index
        step = 0


        # Steps
        while state != reward_location and step <= max_steps:

            # Increases step and a
            step += 1
            learning_curve[trial] = step - min_steps

            # Takes input vector from state_matrix to match state
            input_vector = state_matrix[:, state_index].reshape(states, 1)

            # Calculates Q Values using logsig activation function
            # Logsig used since single layer, ReLU would be better if Deep NN
            q_values = 1 / (1 + np.exp(- weights.dot(input_vector)))
            
            # Epsilon Greedy Policy
            # Takes random action with probability epsilon
            if np.random.rand() > exploration_factor:
                action = np.argmax(q_values)
            # Greedy action
            else: 
                action = np.random.randint(actions)

            # Gets new state based on action taken
            new_state, r_penalty = calculate_new_state(action_effects[action], state, X, Y, walls)
            # Reduces state to index by flattening array with (X,Y) dimensions
            new_state_index = np.ravel_multi_index(new_state, dims=(X,Y), order="F")

            # If this isn't the first step update weights
            if step > 1:
                # Calculate weight update
                dw = learning_rate * (old_reward - old_q_values + discount_factor * q_values[action]) * old_output.dot(old_input.T)

                if SARSA_LAMBDA:
                    # Apply weight update to states in eligibility trace
                    # Done as a batch process
                    dw *= elig_old
                    elig_old *= gamma * lambda_val
                    weights += dw
                
                else:
                    # Update weights as per SARSA
                    weights += dw

            # Create ouput array, set to 0 for action not taken, set to 1 for action taken
            output = np.zeros((actions, 1))
            output[action] = 1

            # Store previous values for future weight update
            old_input = input_vector
            old_output = output
            old_q_values = q_values[action]
            # Since the reward is only recieved at the end
            old_reward = 0
            # Update eligibility trace is SARSA(Lambda) is being used
            if SARSA_LAMBDA:
                elig_old[action][state_index] += 1

            # Update states visited tracking
            if state_index not in states_explored:
                states_explored.append(state_index)

            # Update state
            state = new_state
            state_index = new_state_index

            # Give reward and update weights if the MAX_STEPS hasn't been exceeded and reward found
            if step <= max_steps:
                if state_index == reward_index:
                    dw = learning_rate * ((R + r_penalty) - old_q_values) * old_output * old_input.T
                    weights += dw
            
            # Update weights to reflect max steps exceeded penalty
            else:
                dw = learning_rate * ((MAX_STEP_PENALTY + r_penalty) - old_q_values) * old_output * old_input.T
                weights += dw

    # Plots the preferred directions for each state
    if PLOT_DIRECTIONS:
        calculate_directions_matrix(weights, states, state_matrix, X, Y)

    return learning_curve, len(states_explored)



def calculate_directions_matrix(weights, states, state_matrix, X, Y):
    # Ouputs the preferred directions

    # Array creation for plot
    x_points = np.arange(X)
    y_points = np.arange(Y)
    u = np.zeros((X,Y))
    v = np.zeros((X,Y))

    # Iterates through all states
    for state in range(states):
        # Apply activation function to input vector
        input_vector = state_matrix[:, state].reshape(states, 1)
        q_values = 1 / (1 + np.exp(- weights.dot(input_vector)))
        # Use greedy policy to find action
        action = np.argmax(q_values)
        # Calculate coordinates of states
        x = state // X
        y = state % X

        # Map action onto component vectors, u represents y vector, v represents x vector
        # Unit vectors are used to show preferred direction of travel
        if action == 0:
            u[y][x] = -1
        elif action == 2:
            u[y][x] = 1
        elif action == 1:
            v[y][x] = 1
        else:
            v[y][x] = -1

    # Plot Quiver plot
    fig, ax = plt.subplots()
    q = ax.quiver(x_points,y_points,u,v)
    plt.show()


def calculate_new_state(action, state, X, Y, walls=None):
    # Calculates the new state based on previous state, action, dimensions and precense of walls
    new_state = state
    r_penalty = 0

    # Invoke action
    if action[0] == "N" or action[0] == "S":
        new_state[1] += action[1]
        # Check wall collision
        if new_state in walls:
            r_penalty = -0.1
            new_state[1] -= action[1]

    else:
        new_state[0] += action[1]
        # Check wall collision
        if new_state in walls:
            r_penalty = -0.1
            new_state[0] -= action[1]

    # Check out of bounds
    if new_state[0] < 0:
        new_state[0] = 0
        r_penalty = -0.1
    if new_state[0] >= X:
        new_state[0] = X-1
        r_penalty = -0.1
    if new_state[1] < 0:
        new_state[1] = 0
        r_penalty = -0.1
    if new_state[1] >= Y:
        new_state[1] = Y-1
        r_penalty = -0.1

    return new_state, r_penalty


    

def generate_start(reward_location, X, Y):
    # Creates start location
    start_position = [random.randint(0, X-1), random.randint(0,Y-1)]

    # Stops rewards and start being in the same position
    while reward_location == start_position:
        start_position = [random.randint(0, X-1), random.randint(0,Y-1)]
    
    return start_position



def main():
    # Program parameters
    OUTPUT_STATES_EXPLORED = False
    # Number of experiments
    repetitions = 2
    # Number of trials per experiment
    trials = 1000
    # Maximum number of steps in a walk
    max_steps = 50
    # Learning parameters
    learning_rate = 0.8 
    epsilon = 0.1
    gamma = 0.9
    # Stores states explored per experiment
    states_explored = []
    # Initialise empty learning curve
    learning_curve = np.zeros((repetitions, trials))

    # Run experiments
    for i in range(repetitions):
            learning_curve[i], states_explored_trial = homing(trials, max_steps, learning_rate, gamma, epsilon)
            states_explored.append(states_explored_trial)

    # Output average number of states explored in an experiment
    if OUTPUT_STATES_EXPLORED:
            print("States explores: ", sum(states_explored) / len(states_explored))

    # Calculate means in learning curve
    means = np.mean(learning_curve, axis=0)
    # Calculate standard deviation in learning curve
    errors = np.std(learning_curve, axis = 0) / np.sqrt(repetitions)

    # Applies guassian filter to smooth outputs
    smooth_means = gaussian_filter1d(means, 2)
    smooth_errors = gaussian_filter1d(errors, 2)

    # Produce error bars
    plt.errorbar(np.arange(trials), smooth_means, smooth_errors, 0, elinewidth = 0.1, capsize = 1, alpha =0.2)
    # Plots mean
    plt.plot(smooth_means, label="SARSA")

    # Displays plot
    plt.legend(loc="upper right", fontsize=10)
    plt.xlabel('Trials',fontsize = 16)
    plt.ylabel('Average Steps Away From Optimum',fontsize = 16)
    plt.tick_params(axis = 'both', which='major', labelsize = 14)
    plt.savefig('Sarsa.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()