#!/usr/bin/env python
import gym 
import safety_gym
import safe_rl
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork
from safety_gym.envs.engine import Engine


# Point
# Goal1/2 (60,2)
# Push1/2 (76,2)
# Button1/2 (76,2)

# Robot
# Goal1/2 (72,2)
# Push1/2 (88,2)
# Button1/2 (88,2)

# Doggo
# Goal1/2 (104,12)
# Push1/2 (120,12)
# Button1/2 (120,2)

def main(robot, task, algo, seed, exp_name, cpu, args):

    # Verify experiment
    robot_list = ['point', 'car', 'doggo']
    task_list = ['mygoal1','mygoal4', "mypush1","mypush4", "mygoal1_pillars", "mygoal4_pillars",'goal1', 'goal2', 'button1', 'button2', 'push1', 'push2']
    algo_list = ['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo', 'ppo_adamba', 'trpo_adamba','ppo_adamba_sc','trpo_adamba_sc']
    algo = algo.lower()
    task = task.capitalize()
    robot = robot.capitalize()
    assert algo in algo_list, "Invalid algo"
    assert task.lower() in task_list, "Invalid task"
    assert robot.lower() in robot_list, "Invalid robot"
    # Hyperparameters
    if exp_name == None:
        if "pillars" in task.lower():
            if 'adamba' not in algo.lower():
                if args.cpc:
                    raise NotImplementedError
                exp_name = algo + '_' + robot + task + 'pillars_size-' + str(args.pillars_size) + "_" + "ctrlrange-" + str(args.ctrlrange)
            else:
                if args.index[0:10] == 'projection':
                    if args.cpc:
                        exp_name = algo + '_' +  robot + task + 'pillars_size-' + str(args.pillars_size) + '_' + args.index + '_margin-' + str(args.margin) + "_" + "threshold-" + str(args.threshold) + "_" + "ctrlrange-" + str(args.ctrlrange) + "_" + "cpc-" + str(args.cpc) + "_" + "cpc_coef-" + str(args.cpc_coef)+ "_" + "pre_execute-" + str(args.pre_execute) + "_" + "pre_execute_coef" + "-" + str(args.pre_execute_coef) 
                    else:
                        exp_name = algo + '_' +  robot + task + 'pillars_size-' + str(args.pillars_size) + '_' + args.index + '_margin-' + str(args.margin) + "_" + "threshold-" + str(args.threshold) + "_" + "ctrlrange-" + str(args.ctrlrange) + "_" + "pre_execute-" + str(args.pre_execute) + "_" + "pre_execute_coef" + "-" + str(args.pre_execute_coef) 
                
                    exp_name += "_simulation_fixed"
                elif args.index == 'adaptive':
                    if args.cpc:
                        exp_name = algo + '_' +  robot + task + 'pillars_size-' + str(args.pillars_size) + '_' + args.index + '_k-' + str(args.k) + "_" + "sigma-" + str(args.sigma) + "_" + 'n-' + str(args.n) + "_" + "threshold-" + str(args.threshold) + '_'+ "ctrlrange-" + str(args.ctrlrange) + "_" + "cpc-" + str(args.cpc) + "_" + "cpc_coef-" + str(args.cpc_coef)+ "_" + "pre_execute-" + str(args.pre_execute) + "_" + "pre_execute_coef" + "-" + str(args.pre_execute_coef) 
                    else:
                        exp_name = algo + '_' +  robot + task + 'pillars_size-' + str(args.pillars_size) + '_' + args.index + '_k-' + str(args.k) + "_" + "sigma-" + str(args.sigma) + "_" + 'n-' + str(args.n) + "_" + "threshold-" + str(args.threshold) + '_'+ "ctrlrange-" + str(args.ctrlrange) + "_" + "pre_execute-" + str(args.pre_execute) + "_" + "pre_execute_coef" + "-" + str(args.pre_execute_coef) 

                    exp_name += "_simulation_fixed"
                else:
                    raise NotImplementedError     
        else:
            if 'adamba' not in algo.lower():
                if args.cpc:
                    raise NotImplementedError
                exp_name = algo + '_' + robot + task + 'hazards_size-' + str(args.hazards_size) + "_" + "ctrlrange-" + str(args.ctrlrange)
            else:
                if args.index[0:10] == 'projection':
                    if args.cpc:
                        exp_name = algo + '_' +  robot + task + 'hazards_size-' + str(args.hazards_size) + '_' + args.index + '_margin-' + str(args.margin) + "_" + "threshold-" + str(args.threshold) + "_" + "ctrlrange-" + str(args.ctrlrange) + "_" + "cpc-" + str(args.cpc) + "_" + "cpc_coef-" + str(args.cpc_coef)+ "_" + "pre_execute-" + str(args.pre_execute) + "_" + "pre_execute_coef" + "-" + str(args.pre_execute_coef) 
                    else:
                        exp_name = algo + '_' +  robot + task + 'hazards_size-' + str(args.hazards_size) + '_' + args.index + '_margin-' + str(args.margin) + "_" + "threshold-" + str(args.threshold) + "_" + "ctrlrange-" + str(args.ctrlrange) + "_" + "pre_execute-" + str(args.pre_execute) + "_" + "pre_execute_coef" + "-" + str(args.pre_execute_coef) 
                
                    exp_name += "_simulation_fixed"
                elif args.index == 'adaptive':
                    if args.cpc:
                        exp_name = algo + '_' +  robot + task + 'hazards_size-' + str(args.hazards_size) + '_' + args.index + '_k-' + str(args.k) + "_" + "sigma-" + str(args.sigma) + "_" + 'n-' + str(args.n) + "_" + "threshold-" + str(args.threshold) + '_'+ "ctrlrange-" + str(args.ctrlrange) + "_" + "cpc-" + str(args.cpc) + "_" + "cpc_coef-" + str(args.cpc_coef)+ "_" + "pre_execute-" + str(args.pre_execute) + "_" + "pre_execute_coef" + "-" + str(args.pre_execute_coef) 
                    else:
                        exp_name = algo + '_' +  robot + task + 'hazards_size-' + str(args.hazards_size) + '_' + args.index + '_k-' + str(args.k) + "_" + "sigma-" + str(args.sigma) + "_" + 'n-' + str(args.n) + "_" + "threshold-" + str(args.threshold) + '_'+ "ctrlrange-" + str(args.ctrlrange) + "_" + "pre_execute-" + str(args.pre_execute) + "_" + "pre_execute_coef" + "-" + str(args.pre_execute_coef) 

                    exp_name += "_simulation_fixed"
                else:
                    raise NotImplementedError
    if robot=='Doggo':
        num_steps = 1e8
        steps_per_epoch = 60000
    else:
        num_steps = 1e7
        steps_per_epoch = 30000
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 50
    target_kl = 0.01
    cost_lim = 0

    # Fork for parallelizing
    mpi_fork(cpu)

    # Prepare Logger


    logger_kwargs = setup_logger_kwargs(exp_name, seed)
    
    # Algo and Env
    algo = eval('safe_rl.'+algo)

    if task == "Mygoal1":
        # finite time convergence test suite
        # config = {
        #     'robot_base': 'xmls/point.xml', # dt in xml, default 0.002s for point

        #     # finite time convergence test suite modification
        #     'robot_placements': None,  # Robot placements list (defaults to full extents)
        #     'robot_locations': [[0.0, 0.0]],  # Explicitly place robot XY coordinate
        #     'robot_keepout': 0.0,  # Needs to be set to match the robot XML used
        #     # Hazardous areas
        #     'hazards_placements': None,  # Placements list for hazards (defaults to full extents)
        #     'hazards_locations': [[-0.3, -0.3]],  # Fixed locations to override placements
        #     'hazards_keepout': 0.0,  # Radius of hazard keepout for placement
        #     'hazards_num': 1,
        #     'hazards_size': 0.5,

        #     'task': 'goal',
        #     'observation_flatten': True,  # Flatten observation into a vector
        #     'observe_sensors': True,  # Observe all sensor data from simulator
        #     # Sensor observations
        #     # Specify which sensors to add to observation space
        #     'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
        #     'sensors_hinge_joints': True,  # Observe named joint position / velocity sensors
        #     'sensors_ball_joints': True,  # Observe named balljoint position / velocity sensors
        #     'sensors_angle_components': True,  # Observe sin/cos theta instead of theta

        #     #observe goal/box/...
        #     'observe_goal_dist': False,  # Observe the distance to the goal
        #     'observe_goal_comp': False,  # Observe a compass vector to the goal
        #     'observe_goal_lidar': True,  # Observe the goal with a lidar sensor
        #     'observe_box_comp': False,  # Observe the box with a compass
        #     'observe_box_lidar': False,  # Observe the box with a lidar
        #     'observe_circle': False,  # Observe the origin with a lidar
        #     'observe_remaining': False,  # Observe the fraction of steps remaining
        #     'observe_walls': False,  # Observe the walls with a lidar space
        #     'observe_hazards': True,  # Observe the vector from agent to hazards
        #     'observe_vases': True,  # Observe the vector from agent to vases
        #     'observe_pillars': False,  # Lidar observation of pillar object positions
        #     'observe_buttons': False,  # Lidar observation of button object positions
        #     'observe_gremlins': False,  # Gremlins are observed with lidar-like space
        #     'observe_vision': False,  # Observe vision from the robot

        #     # Constraints - flags which can be turned on
        #     # By default, no constraints are enabled, and all costs are indicator functions.
        #     'constrain_hazards': True,  # Constrain robot from being in hazardous areas
        #     'constrain_vases': False,  # Constrain frobot from touching objects
        #     'constrain_pillars': False,  # Immovable obstacles in the environment
        #     'constrain_buttons': False,  # Penalize pressing incorrect buttons
        #     'constrain_gremlins': False,  # Moving objects that must be avoided
        #     # cost discrete/continuous. As for AdamBA, I guess continuous cost is more suitable.
        #     'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

        #     #lidar setting
        #     'lidar_max_dist': None, # Maximum distance for lidar sensitivity (if None, exponential distance)
        #     'lidar_num_bins': 16,
        #     #num setting

        #     'vases_num': 0,

        #     # dt perhaps?

        #     # Frameskip is the number of physics simulation steps per environment step
        #     # Frameskip is sampled as a binomial distribution
        #     # For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip)
        #     'frameskip_binom_n': 10,  # Number of draws trials in binomial distribution (max frameskip) 
        #     'frameskip_binom_p': 1.0  # Probability of trial return (controls distribution)
        # }

        config = {
            'robot_base': 'xmls/point.xml', # dt in xml, default 0.002s for point
            'task': 'goal',
            'observation_flatten': True,  # Flatten observation into a vector
            'observe_sensors': True,  # Observe all sensor data from simulator
            # Sensor observations
            # Specify which sensors to add to observation space
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
            'sensors_hinge_joints': True,  # Observe named joint position / velocity sensors
            'sensors_ball_joints': True,  # Observe named balljoint position / velocity sensors
            'sensors_angle_components': True,  # Observe sin/cos theta instead of theta

            #observe goal/box/...
            'observe_goal_dist': False,  # Observe the distance to the goal
            'observe_goal_comp': False,  # Observe a compass vector to the goal
            'observe_goal_lidar': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': False,  # Observe the box with a compass
            'observe_box_lidar': False,  # Observe the box with a lidar
            'observe_circle': False,  # Observe the origin with a lidar
            'observe_remaining': False,  # Observe the fraction of steps remaining
            'observe_walls': False,  # Observe the walls with a lidar space
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'observe_vases': True,  # Observe the vector from agent to vases
            'observe_pillars': False,  # Lidar observation of pillar object positions
            'observe_buttons': False,  # Lidar observation of button object positions
            'observe_gremlins': False,  # Gremlins are observed with lidar-like space
            'observe_vision': False,  # Observe vision from the robot

            # Constraints - flags which can be turned on
            # By default, no constraints are enabled, and all costs are indicator functions.
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_vases': False,  # Constrain frobot from touching objects
            'constrain_pillars': False,  # Immovable obstacles in the environment
            'constrain_buttons': False,  # Penalize pressing incorrect buttons
            'constrain_gremlins': False,  # Moving objects that must be avoided
            # cost discrete/continuous. As for AdamBA, I guess continuous cost is more suitable.
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            #lidar setting
            'lidar_max_dist': None, # Maximum distance for lidar sensitivity (if None, exponential distance)
            'lidar_num_bins': 16,
            #num setting
            'hazards_num': 1,
            'hazards_size': args.hazards_size,
            'vases_num': 0,



            # Frameskip is the number of physics simulation steps per environment step
            # Frameskip is sampled as a binomial distribution
            # For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip)
            'frameskip_binom_n': 10,  # Number of draws trials in binomial distribution (max frameskip) 
            'frameskip_binom_p': 1.0  # Probability of trial return (controls distribution)
        }

    if task == "Mygoal4":
        config = {
            'robot_base': 'xmls/point.xml', # dt in xml, default 0.002s for point
            'task': 'goal',
            'observation_flatten': True,  # Flatten observation into a vector
            'observe_sensors': True,  # Observe all sensor data from simulator
            # Sensor observations
            # Specify which sensors to add to observation space
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
            'sensors_hinge_joints': True,  # Observe named joint position / velocity sensors
            'sensors_ball_joints': True,  # Observe named balljoint position / velocity sensors
            'sensors_angle_components': True,  # Observe sin/cos theta instead of theta

            #observe goal/box/...
            'observe_goal_dist': False,  # Observe the distance to the goal
            'observe_goal_comp': False,  # Observe a compass vector to the goal
            'observe_goal_lidar': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': False,  # Observe the box with a compass
            'observe_box_lidar': False,  # Observe the box with a lidar
            'observe_circle': False,  # Observe the origin with a lidar
            'observe_remaining': False,  # Observe the fraction of steps remaining
            'observe_walls': False,  # Observe the walls with a lidar space
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'observe_vases': True,  # Observe the vector from agent to vases
            'observe_pillars': False,  # Lidar observation of pillar object positions
            'observe_buttons': False,  # Lidar observation of button object positions
            'observe_gremlins': False,  # Gremlins are observed with lidar-like space
            'observe_vision': False,  # Observe vision from the robot

            # Constraints - flags which can be turned on
            # By default, no constraints are enabled, and all costs are indicator functions.
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_vases': False,  # Constrain frobot from touching objects
            'constrain_pillars': False,  # Immovable obstacles in the environment
            'constrain_buttons': False,  # Penalize pressing incorrect buttons
            'constrain_gremlins': False,  # Moving objects that must be avoided
            # cost discrete/continuous. As for AdamBA, I guess continuous cost is more suitable.
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            #lidar setting
            'lidar_max_dist': None, # Maximum distance for lidar sensitivity (if None, exponential distance)
            'lidar_num_bins': 16,
            #num setting
            'hazards_num': 4,
            'hazards_size': args.hazards_size,
            'vases_num': 0,



            # Frameskip is the number of physics simulation steps per environment step
            # Frameskip is sampled as a binomial distribution
            # For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip)
            'frameskip_binom_n': 10,  # Number of draws trials in binomial distribution (max frameskip) 
            'frameskip_binom_p': 1.0  # Probability of trial return (controls distribution)
        }

    if task == "Mypush4":


        config = {
            'robot_base': 'xmls/point.xml', # dt in xml, default 0.002s for point
            'task': 'push',
            'box_size': 0.2,
            'box_null_dist': 0,

            'observation_flatten': True,  # Flatten observation into a vector
            'observe_sensors': True,  # Observe all sensor data from simulator
            # Sensor observations
            # Specify which sensors to add to observation space
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
            'sensors_hinge_joints': True,  # Observe named joint position / velocity sensors
            'sensors_ball_joints': True,  # Observe named balljoint position / velocity sensors
            'sensors_angle_components': True,  # Observe sin/cos theta instead of theta

            #observe goal/box/...
            'observe_goal_dist': False,  # Observe the distance to the goal
            'observe_goal_comp': False,  # Observe a compass vector to the goal
            'observe_goal_lidar': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': False,  # Observe the box with a compass
            'observe_box_lidar': True,  # Observe the box with a lidar
            'observe_circle': False,  # Observe the origin with a lidar
            'observe_remaining': False,  # Observe the fraction of steps remaining
            'observe_walls': False,  # Observe the walls with a lidar space
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'observe_vases': True,  # Observe the vector from agent to vases
            'observe_pillars': False,  # Lidar observation of pillar object positions
            'observe_buttons': False,  # Lidar observation of button object positions
            'observe_gremlins': False,  # Gremlins are observed with lidar-like space
            'observe_vision': False,  # Observe vision from the robot

            # Constraints - flags which can be turned on
            # By default, no constraints are enabled, and all costs are indicator functions.
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_vases': False,  # Constrain frobot from touching objects
            'constrain_pillars': False,  # Immovable obstacles in the environment
            'constrain_buttons': False,  # Penalize pressing incorrect buttons
            'constrain_gremlins': False,  # Moving objects that must be avoided
            # cost discrete/continuous. As for AdamBA, I guess continuous cost is more suitable.
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            #lidar setting
            'lidar_max_dist': None, # Maximum distance for lidar sensitivity (if None, exponential distance)
            'lidar_num_bins': 16,
            #num setting
            'hazards_num': 4,
            'hazards_size': args.hazards_size,
            'vases_num': 0,



            # Frameskip is the number of physics simulation steps per environment step
            # Frameskip is sampled as a binomial distribution
            # For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip)
            'frameskip_binom_n': 10,  # Number of draws trials in binomial distribution (max frameskip) 
            'frameskip_binom_p': 1.0  # Probability of trial return (controls distribution)
        }

    if task == "Mypush1":


        config = {
            'robot_base': 'xmls/point.xml', # dt in xml, default 0.002s for point
            'task': 'push',
            'box_size': 0.2,
            'box_null_dist': 0,

            'observation_flatten': True,  # Flatten observation into a vector
            'observe_sensors': True,  # Observe all sensor data from simulator
            # Sensor observations
            # Specify which sensors to add to observation space
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
            'sensors_hinge_joints': True,  # Observe named joint position / velocity sensors
            'sensors_ball_joints': True,  # Observe named balljoint position / velocity sensors
            'sensors_angle_components': True,  # Observe sin/cos theta instead of theta

            #observe goal/box/...
            'observe_goal_dist': False,  # Observe the distance to the goal
            'observe_goal_comp': False,  # Observe a compass vector to the goal
            'observe_goal_lidar': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': False,  # Observe the box with a compass
            'observe_box_lidar': True,  # Observe the box with a lidar
            'observe_circle': False,  # Observe the origin with a lidar
            'observe_remaining': False,  # Observe the fraction of steps remaining
            'observe_walls': False,  # Observe the walls with a lidar space
            'observe_hazards': True,  # Observe the vector from agent to hazards
            'observe_vases': True,  # Observe the vector from agent to vases
            'observe_pillars': False,  # Lidar observation of pillar object positions
            'observe_buttons': False,  # Lidar observation of button object positions
            'observe_gremlins': False,  # Gremlins are observed with lidar-like space
            'observe_vision': False,  # Observe vision from the robot

            # Constraints - flags which can be turned on
            # By default, no constraints are enabled, and all costs are indicator functions.
            'constrain_hazards': True,  # Constrain robot from being in hazardous areas
            'constrain_vases': False,  # Constrain frobot from touching objects
            'constrain_pillars': False,  # Immovable obstacles in the environment
            'constrain_buttons': False,  # Penalize pressing incorrect buttons
            'constrain_gremlins': False,  # Moving objects that must be avoided
            # cost discrete/continuous. As for AdamBA, I guess continuous cost is more suitable.
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            #lidar setting
            'lidar_max_dist': None, # Maximum distance for lidar sensitivity (if None, exponential distance)
            'lidar_num_bins': 16,
            #num setting
            'hazards_num': 1,
            'hazards_size': args.hazards_size,
            'vases_num': 0,



            # Frameskip is the number of physics simulation steps per environment step
            # Frameskip is sampled as a binomial distribution
            # For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip)
            'frameskip_binom_n': 10,  # Number of draws trials in binomial distribution (max frameskip) 
            'frameskip_binom_p': 1.0  # Probability of trial return (controls distribution)
        }

    if task == "Mygoal4_pillars":
        config = {
            'robot_base': 'xmls/point.xml', # dt in xml, default 0.002s for point
            'task': 'goal',
            'observation_flatten': True,  # Flatten observation into a vector
            'observe_sensors': True,  # Observe all sensor data from simulator
            # Sensor observations
            # Specify which sensors to add to observation space
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
            'sensors_hinge_joints': True,  # Observe named joint position / velocity sensors
            'sensors_ball_joints': True,  # Observe named balljoint position / velocity sensors
            'sensors_angle_components': True,  # Observe sin/cos theta instead of theta

            #observe goal/box/...
            'observe_goal_dist': False,  # Observe the distance to the goal
            'observe_goal_comp': False,  # Observe a compass vector to the goal
            'observe_goal_lidar': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': False,  # Observe the box with a compass
            'observe_box_lidar': False,  # Observe the box with a lidar
            'observe_circle': False,  # Observe the origin with a lidar
            'observe_remaining': False,  # Observe the fraction of steps remaining
            'observe_walls': False,  # Observe the walls with a lidar space
            'observe_hazards': False,  # Observe the vector from agent to hazards
            'observe_vases': False,  # Observe the vector from agent to vases
            'observe_pillars': True,  # Lidar observation of pillar object positions
            'observe_buttons': False,  # Lidar observation of button object positions
            'observe_gremlins': False,  # Gremlins are observed with lidar-like space
            'observe_vision': False,  # Observe vision from the robot

            # Constraints - flags which can be turned on
            # By default, no constraints are enabled, and all costs are indicator functions.
            'constrain_hazards': False,  # Constrain robot from being in hazardous areas
            'constrain_vases': False,  # Constrain frobot from touching objects
            'constrain_pillars': True,  # Immovable obstacles in the environment
            'constrain_buttons': False,  # Penalize pressing incorrect buttons
            'constrain_gremlins': False,  # Moving objects that must be avoided
            # cost discrete/continuous. As for AdamBA, I guess continuous cost is more suitable.
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            #lidar setting
            'lidar_max_dist': None, # Maximum distance for lidar sensitivity (if None, exponential distance)
            'lidar_num_bins': 16,
            #num setting
            'hazards_num': 0,
            'hazards_size': args.hazards_size,
            'vases_num': 0,

            # Pillars (immovable obstacles we should not touch)

            # 'robot_keepout': 0.4,

            'pillars_num': 4,  # Number of pillars in the world
            'pillars_placements': None,  # Pillars placements list (defaults to full extents)
            # 'pillars_locations': [],  # Fixed locations to override placements
            'pillars_keepout': 0.4,  # Radius for placement of pillars
            'pillars_size': args.pillars_size,  # Half-size (radius) of pillar objects
            'pillars_height': 0.5,  # Half-height of pillars geoms
            'pillars_cost': 1.0,  # Cost (per step) for being in contact with a pillar




            # Frameskip is the number of physics simulation steps per environment step
            # Frameskip is sampled as a binomial distribution
            # For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip)
            'frameskip_binom_n': 10,  # Number of draws trials in binomial distribution (max frameskip) 
            'frameskip_binom_p': 1.0  # Probability of trial return (controls distribution)
        }

    if task == "Mygoal1_pillars":
        config = {
            'robot_base': 'xmls/point.xml', # dt in xml, default 0.002s for point
            'task': 'goal',
            'observation_flatten': True,  # Flatten observation into a vector
            'observe_sensors': True,  # Observe all sensor data from simulator
            # Sensor observations
            # Specify which sensors to add to observation space
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
            'sensors_hinge_joints': True,  # Observe named joint position / velocity sensors
            'sensors_ball_joints': True,  # Observe named balljoint position / velocity sensors
            'sensors_angle_components': True,  # Observe sin/cos theta instead of theta

            #observe goal/box/...
            'observe_goal_dist': False,  # Observe the distance to the goal
            'observe_goal_comp': False,  # Observe a compass vector to the goal
            'observe_goal_lidar': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': False,  # Observe the box with a compass
            'observe_box_lidar': False,  # Observe the box with a lidar
            'observe_circle': False,  # Observe the origin with a lidar
            'observe_remaining': False,  # Observe the fraction of steps remaining
            'observe_walls': False,  # Observe the walls with a lidar space
            'observe_hazards': False,  # Observe the vector from agent to hazards
            'observe_vases': False,  # Observe the vector from agent to vases
            'observe_pillars': True,  # Lidar observation of pillar object positions
            'observe_buttons': False,  # Lidar observation of button object positions
            'observe_gremlins': False,  # Gremlins are observed with lidar-like space
            'observe_vision': False,  # Observe vision from the robot

            # Constraints - flags which can be turned on
            # By default, no constraints are enabled, and all costs are indicator functions.
            'constrain_hazards': False,  # Constrain robot from being in hazardous areas
            'constrain_vases': False,  # Constrain frobot from touching objects
            'constrain_pillars': True,  # Immovable obstacles in the environment
            'constrain_buttons': False,  # Penalize pressing incorrect buttons
            'constrain_gremlins': False,  # Moving objects that must be avoided
            # cost discrete/continuous. As for AdamBA, I guess continuous cost is more suitable.
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            #lidar setting
            'lidar_max_dist': None, # Maximum distance for lidar sensitivity (if None, exponential distance)
            'lidar_num_bins': 16,
            #num setting
            'hazards_num': 0,
            'hazards_size': args.hazards_size,
            'vases_num': 0,

            # Pillars (immovable obstacles we should not touch)

            # 'robot_keepout': 0.4,

            'pillars_num': 1,  # Number of pillars in the world
            'pillars_placements': None,  # Pillars placements list (defaults to full extents)
            # 'pillars_locations': [],  # Fixed locations to override placements
            'pillars_keepout': 0.4,  # Radius for placement of pillars
            'pillars_size': args.pillars_size,  # Half-size (radius) of pillar objects
            'pillars_height': 0.5,  # Half-height of pillars geoms
            'pillars_cost': 1.0,  # Cost (per step) for being in contact with a pillar




            # Frameskip is the number of physics simulation steps per environment step
            # Frameskip is sampled as a binomial distribution
            # For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip)
            'frameskip_binom_n': 10,  # Number of draws trials in binomial distribution (max frameskip) 
            'frameskip_binom_p': 1.0  # Probability of trial return (controls distribution)
        }

    algo(env_fn=lambda: Engine(config),
            ac_kwargs=dict(
                hidden_sizes=(256, 256),
            ),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            save_freq=save_freq,
            target_kl=target_kl,
            cost_lim=cost_lim,
            seed=seed,
            logger_kwargs=logger_kwargs,
            margin=args.margin,
            threshold=args.threshold,
            ctrlrange=args.ctrlrange,
            # adaptive safety index,
            k = args.k,
            n = args.n,
            sigma = args.sigma,
            cpc = args.cpc,
            cpc_coef = args.cpc_coef,
            pre_execute = args.pre_execute,
            pre_execute_coef = args.pre_execute_coef
            )

    




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='Point')
    parser.add_argument('--task', type=str, default='myGoal1')
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--index', type=str, default='adaptive') # ['adaptive', 'projection_max', 'projection_argmin']
    parser.add_argument('--k', type=float, default=2.0)
    parser.add_argument('--n', type=float, default=2.0)
    parser.add_argument('--hazards_size', type=float, default=0.30)
    parser.add_argument('--pillars_size', type=float, default=0.30)
    parser.add_argument('--sigma', type=float, default=0.04)
    parser.add_argument('--margin', type=float, default=0.4)
    parser.add_argument('--ctrlrange', type=float, default=10.0)
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--pre_execute', type=bool, default=False)
    parser.add_argument('--pre_execute_coef', type=float, default=0.0)
    parser.add_argument('--cpc', type=bool, default=False)
    parser.add_argument('--cpc_coef', type=float, default=0.01)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--cpu', type=int, default=1)

    args = parser.parse_args()
    exp_name = args.exp_name if not(args.exp_name=='') else None
    main(args.robot, args.task, args.algo, args.seed, exp_name, args.cpu, args)