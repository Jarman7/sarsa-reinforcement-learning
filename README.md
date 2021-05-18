# sarsa-reinforcement-learning
## Setup
There main program is ran through the ```main()``` function. Parameters can be set here for:
* repetitions - The number of experiments
* trials - The number of trials per experiement
* max_steps - The maximum number of steps per walk
* learning_rate
* epsilon - For epsilon-greedy policy, 0 means just greedy
* gamma - discount factor
* OUTPUT_STATES_EXPLORED - Boolean determines if the average number of states explored in the experiments is output

Optional parameters to pass to ```homing()``` function, these default to ```0``` or ```False```:
* lambda_val - The lambda value used in SARSA(Lambda) learning
* SARSA_LAMBDA - Boolean, determines if SARSA(Lambda) learning is used
* PLOT_DIRECTIONS - Boolean, determines if preferred directions for each state at the end of the experiment

The ```main()``` function is automatically ran.