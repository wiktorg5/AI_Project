import numpy as np
import parking_model as pm


def reward(param_phis, state, if_collision, if_stop):
    value = 0
    x = state[0]
    y = state[1]
    alfa = state[2]
    dist_xy_sq = x * x + y * y
    alfa_zred = 0
    if param_phis.if_side_parking_place:
        if np.abs(alfa) > np.pi / 2:
            alfa_zred = np.pi - np.abs(alfa)
        else:
            alfa_zred = np.abs(alfa)
    else:
        alfa_zred = np.abs(np.abs(alfa) - np.pi / 2)

    alfa_zred = alfa_zred / (dist_xy_sq + 0.5)

    dist_eval = 1 / (dist_xy_sq + 0.5) - 1
    angle_eval = alfa_zred - 0.5

    # if V==0 reward is calculated based on a distance

    if if_collision:
        value = -1
    elif if_stop:
        value = min(dist_eval, angle_eval)
    else:
        value = 0

    return value


def choose_action(param_phis, state):
    # here the learned strategy should be used in pure exploitation
    # the strategy can be, for example, represented by an approximator of the utility function
    # ..........................................
    # ..........................................

    angle = -np.pi / 8  # steering angle (for now)
    V = -param_phis.Vmod  # speed (for now)
    return angle, V


def park_test(param_phis, initial_state):
    pm.park_save("param.txt", param_phis)
    phist = open('history.txt', 'w')
    num_of_initial_states, lparam = initial_state.shape
    avg_sum_of_rewards = 0
    num_of_steps = 0
    for episode in range(num_of_initial_states):
        # We choose the starting state:
        init_state_no = episode % num_of_initial_states
        state = initial_state[init_state_no, :]

        step = 0
        if_collision = False
        if_stop = False
        sum_of_rewards_in_episode = 0
        while if_stop == False:
            step = step + 1

            # We determine actions a (angle + direction of motion) in the state state according to the learned strategy:
            angle, V = choose_action(param_phis, state)

            # saveing the step of history :
            # phist.write(str(episode + 1) + "  " + str(step) + "  " + str(state[0]) + "  " + str(state[1]) + "  " + str(state[2]) + "  " + str(angle) + "  " + str(V) + "\n")
            phist.write(
                "%d %d %.4f %.4f %.4f %.4f %.4f\n" % ((episode + 1), step, state[0], state[1], state[2], angle, V))
            # new state determination:
            new_state, rotation_center, if_collision = pm.model_of_car(state, angle, V, param_phis)

            if (if_collision) | (step >= param_phis.max_number_of_steps):
                if_stop = True

            R = reward(param_phis, new_state, if_collision, if_stop)
            sum_of_rewards_in_episode += R

            state = new_state

        avg_sum_of_rewards = avg_sum_of_rewards + sum_of_rewards_in_episode / num_of_initial_states
        num_of_steps = num_of_steps + step
        print("in %d episode sum of rewards = %g, num of steps = %d" % (episode, sum_of_rewards_in_episode, step))

    print("average sum of rewards in episode = %g" % (avg_sum_of_rewards))
    print("average number of steps = %g" % (num_of_steps / num_of_initial_states))
    phist.close()


# training procedure proper for reinforcement learning with approximation of action values:
def park_train():
    liczba_epizodow = 2000
    alfa = 0.001  # learning rate (may be a function of time)
    epsylon = 0.1  # exploration factor (may be a function of time)

    initial_state = np.array([[9.1, 4.6, 0], [6.3, 5.06, 0], [9.6, 3.15, 0], [7.3, 5.75, 0], [10.1, 6.21, 0]],
                             dtype=float)
    num_of_initial_states, lparam = initial_state.shape

    param_phis = pm.GlobalVar()  # phisical parameters of a parking and a car

    # initiation of coding, determination of the number of parameters (weights):
    # ........................................................
    # ........................................................

    # weight vector initialization:
    liczba_wag = 1000  # for now to start a program
    w = np.zeros(liczba_wag)

    for episode in range(liczba_epizodow):
        # Initial state choosing:
        init_state_no = episode % num_of_initial_states
        state = initial_state[init_state_no, :]

        step = 0
        if_collision = False
        if_stop = False
        while if_stop == False:
            step = step + 1

            # We determine actions a (angle + direction of motion) in the state state, 
            # taking into account exploration (e.g. in reinforcement learning, 
            # the epsylon-greedy or softmax method)
            # ........................................................
            # ........................................................
            angle = np.pi / 8  # for now
            V = param_phis.Vmod;  # for now
            # determination of a new state:
            new_state, rotation_center, if_collision = pm.model_of_car(state, angle, V, param_phis)

            if (if_collision) | (step >= param_phis.max_number_of_steps):
                if_stop = True

            R = reward(param_phis, new_state, if_collision, if_stop)

            # We update the Q values for the current state and selected action:
            # ........................................................
            # ........................................................
            # w = w + ...

            state = new_state

        # test with generating history to a file:
        if episode % 1000 == 0:
            print("episode %d\n" % episode)
            park_test(param_phis, initial_state)


park_train()
