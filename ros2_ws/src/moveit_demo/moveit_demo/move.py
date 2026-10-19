#!/usr/bin/env python3

import rclpy
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from rclpy.logging import get_logger
from moveit.planning import MoveItPy, PlanRequestParameters
from geometry_msgs.msg import PoseStamped
from moveit.core.robot_state import RobotState
from moveit.core.kinematic_constraints import construct_joint_constraint
from moveit_msgs.msg import AttachedCollisionObject
import json

BASE_FRAME = "base_link"

def load_grasp_point():
    pkg = get_package_share_directory("moveit_demo") # Locate grasp point file
    path = os.path.join(pkg, "config", "grasp.json")

    with open(path, "r") as f:
        d = json.load(f)

    return d["x"], d["y"], d["z"] # Return grasp position

def add_object_to_scene(robot, logger, grasp_xyz, extent):
    """Add the reconstructed object to the planning scene."""
    psm = robot.get_planning_scene_monitor()
    with psm.read_write() as scene: 
        obj = CollisionObject() # Create collision object
        obj.header.frame_id = BASE_FRAME
        obj.id = "reconstructed_object"

        primitive = SolidPrimitive() # Define box dimensions
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = list(extent)

        pose = Pose() # Position object at grasp location
        pose.position.x, pose.position.y, pose.position.z = grasp_xyz
        pose.orientation.w = 1.0

        obj.primitives.append(primitive)
        obj.primitive_poses.append(pose)
        obj.operation = CollisionObject.ADD

        scene.apply_collision_object(obj) # Add object to planning scene
        scene.current_state.update()
    logger.info(f"Added collision object 'reconstructed_object' at {grasp_xyz}, size {extent}")

def attach_object(robot, logger):
    psm = robot.get_planning_scene_monitor()
    with psm.read_write() as scene:
        aco = AttachedCollisionObject() # Attach object to end-effector
        aco.link_name = "tool0"
        aco.object.id = "reconstructed_object"
        aco.object.operation = CollisionObject.ADD

        scene.process_attached_collision_object(aco) # Update planning scene
        scene.current_state.update()

    logger.info("Attached object to tool0")


def detach_object(robot, logger):
    psm = robot.get_planning_scene_monitor()
    with psm.read_write() as scene:
        aco = AttachedCollisionObject() # Remove object from end-effector
        aco.link_name = "tool0"
        aco.object.id = "reconstructed_object"
        aco.object.operation = CollisionObject.REMOVE

        scene.process_attached_collision_object(aco)
        scene.current_state.update()

    logger.info("Detached object")

def plan_and_execute(robot, arm, group, logger, vel=0.4, accel=0.4):
    params = PlanRequestParameters(robot, group)  # Configure planner
    params.planning_pipeline = "ompl"
    params.max_velocity_scaling_factor = vel
    params.max_acceleration_scaling_factor = accel

    result = arm.plan(params) # Plan motion

    if result:
        robot.execute(result.trajectory, controllers=[]) # Execute trajectory
        logger.info("Motion executed successfully.")
        return True

    logger.error("Motion planning failed.")
    return False


def main():
    """Execute the pick-and-place task."""
    rclpy.init()
    logger = get_logger("moveit_demo")

    # Initialise robot and planning component
    robot = MoveItPy(node_name="moveit_demo")
    arm = robot.get_planning_component("ur_manipulator")
    group = "ur_manipulator"

    # Joint Space Planning
    model = robot.get_robot_model()
    state = RobotState(model)

    # joint configuration positions arm forward and extended
    joint_goal = [0.0, -0.6, 1.0, -0.4, 0.0, 0.0]
    state.set_joint_group_positions(group, joint_goal)

    joint_constraint = construct_joint_constraint(
        robot_state=state,
        joint_model_group=model.get_joint_model_group(group),
    )

    arm.set_goal_state(motion_plan_constraints=[joint_constraint])
    if not plan_and_execute(robot, arm, group, logger):
        return
    

    gx, gy, gz = load_grasp_point()
    extent = (0.05, 0.03, 0.01)   # approximate book dimensions
    add_object_to_scene(robot, logger, (gx, gy, gz), extent)

    # Cartesian Pose Planning 
    pose_goal = PoseStamped()
    pose_goal.header.frame_id = "base_link"
    pose_goal.pose.position.x = gx
    pose_goal.pose.position.y = gy
    pose_goal.pose.position.z = gz + 0.15  # approach from above
    pose_goal.pose.orientation.x = 1.0
    pose_goal.pose.orientation.y = 0.0
    pose_goal.pose.orientation.z = 0.0
    pose_goal.pose.orientation.w = 0.0

    arm.set_start_state_to_current_state()
    arm.set_goal_state(pose_stamped_msg=pose_goal, pose_link="tool0")

    if not plan_and_execute(robot, arm, group, logger):
        return

    # descend to grasp
    arm.set_start_state_to_current_state()
    pose_goal.pose.position.z = gz + 0.05
    arm.set_goal_state(pose_stamped_msg=pose_goal, pose_link="tool0")
    if not plan_and_execute(robot, arm, group, logger):
        return
    
    # grasp
    attach_object(robot, logger)

    #lift
    arm.set_start_state_to_current_state()
    pose_goal.pose.position.z = gz + 0.20
    arm.set_goal_state(pose_stamped_msg=pose_goal, pose_link="tool0")
    if not plan_and_execute(robot, arm, group, logger):
        return

    #place
    arm.set_start_state_to_current_state()
    pose_goal.pose.position.x = 0.30
    pose_goal.pose.position.y = 0.30
    pose_goal.pose.position.z = 0.20

    arm.set_goal_state(pose_stamped_msg=pose_goal, pose_link="tool0")
    if not plan_and_execute(robot, arm, group, logger):
        return
    detach_object(robot, logger)

    # Return to Home State 
    arm.set_start_state_to_current_state()
    arm.set_goal_state(configuration_name="home")
    if not plan_and_execute(robot, arm, group, logger):
        return

    logger.info("MoveIt demo complete.")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
