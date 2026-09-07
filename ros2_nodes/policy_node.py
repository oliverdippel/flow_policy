"""Milestone 8 -- ROS2 wrapper (STUB, not run against a real ROS2 install).

`rclpy` isn't installed in this environment (no `ros2` CLI, no `/opt/ros`,
`import rclpy` fails) -- per the plan, that means documenting the intended
node/topic design rather than skipping it silently, not skipping the
milestone. This file is a plausible, complete implementation written against
the `rclpy` API, but it has never been executed or imported successfully.
Treat it as a design document with real code in it, not a tested artifact.

Topic design:
  subscribes  /pusht/observation  (std_msgs/Float32MultiArray, len 5) -- the
              same (agent_x, agent_y, block_x, block_y, block_angle) state
              vector the policy was trained on.
  subscribes  /pusht/instruction  (std_msgs/String) -- one of the two fixed
              instruction strings in `flow_policy.envs.TASK_VARIANTS`.
  publishes   /pusht/action       (std_msgs/Float32MultiArray, len 2) -- the
              agent's next target (x, y) position.

Runs Milestone 6's `RecedingHorizonController` internally so the replanning
logic (predict an 8-step chunk, execute 4, replan) is identical to the
Python-only rollout in `flow_policy.rollout` -- this node is purely a
transport layer around it, not a second implementation of the control logic.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from flow_policy.envs import LEFT_TARGET, RIGHT_TARGET
from flow_policy.rollout import RecedingHorizonController, load_policy_from_checkpoint

DEFAULT_CHECKPOINT = "checkpoints/policy_epoch8000.pt"
CONTROL_RATE_HZ = 10.0  # matches gym_pusht's control_hz


class PolicyNode(Node):
    def __init__(self):
        super().__init__("flow_policy_node")
        self.declare_parameter("checkpoint_path", DEFAULT_CHECKPOINT)
        checkpoint_path = self.get_parameter("checkpoint_path").get_parameter_value().string_value

        instructions = [LEFT_TARGET.instruction, RIGHT_TARGET.instruction]
        policy = load_policy_from_checkpoint(checkpoint_path, instructions)
        self._controller = RecedingHorizonController(policy)
        self._controller.reset()

        self._latest_obs = None
        self._latest_instruction = None
        self._t = 0

        self.create_subscription(Float32MultiArray, "/pusht/observation", self._on_observation, 10)
        self.create_subscription(String, "/pusht/instruction", self._on_instruction, 10)
        self._action_pub = self.create_publisher(Float32MultiArray, "/pusht/action", 10)
        self.create_timer(1.0 / CONTROL_RATE_HZ, self._on_timer)

    def _on_observation(self, msg: Float32MultiArray) -> None:
        self._latest_obs = msg.data  # expects len-5 (agent_x, agent_y, block_x, block_y, block_angle)

    def _on_instruction(self, msg: String) -> None:
        self._latest_instruction = msg.data

    def _on_timer(self) -> None:
        if self._latest_obs is None or self._latest_instruction is None:
            return  # no observation/instruction yet -- nothing to act on

        obs = np.asarray(self._latest_obs, dtype=np.float32)
        action = self._controller.act(obs, self._latest_instruction, self._t)
        self._t += 1

        msg = Float32MultiArray()
        msg.data = [float(action[0]), float(action[1])]
        self._action_pub.publish(msg)


def main():
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
