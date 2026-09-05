"""Milestone 8 -- ROS2 wrapper (STUB, not run against a real ROS2 install).

See `policy_node.py` for why this is a documented design rather than a
tested artifact. This node bridges the `gym-pusht` environment itself to the
same topics `policy_node.py` talks on, so the full loop -- env -> topic ->
policy -> topic -> env -- runs through ROS2 message passing instead of a
direct Python function call, proving the deployment interface rather than
just the standalone inference wrapper.

Design: publishes the current observation and instruction once at startup
(the instruction never changes mid-episode) and again after every step;
stepping is driven by incoming `/pusht/action` messages rather than a timer,
so the env advances exactly once per action the policy node produces.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from flow_policy.envs import LEFT_TARGET, make_variant_env

EPISODE_VARIANT = LEFT_TARGET  # which fixed task variant this bridge instance runs


class PushTBridgeNode(Node):
    def __init__(self):
        super().__init__("pusht_bridge_node")
        self._env = make_variant_env(EPISODE_VARIANT, render_mode="rgb_array")
        self._obs, _info = self._env.reset()

        self._obs_pub = self.create_publisher(Float32MultiArray, "/pusht/observation", 10)
        self._instruction_pub = self.create_publisher(String, "/pusht/instruction", 10)
        self.create_subscription(Float32MultiArray, "/pusht/action", self._on_action, 10)

        self._publish_instruction()
        self._publish_observation()

    def _publish_observation(self) -> None:
        msg = Float32MultiArray()
        msg.data = [float(x) for x in self._obs]
        self._obs_pub.publish(msg)

    def _publish_instruction(self) -> None:
        msg = String()
        msg.data = EPISODE_VARIANT.instruction
        self._instruction_pub.publish(msg)

    def _on_action(self, msg: Float32MultiArray) -> None:
        action = np.array(msg.data, dtype=np.float32)
        self._obs, reward, terminated, truncated, info = self._env.step(action)
        self._publish_observation()
        if terminated or truncated:
            self.get_logger().info(f"episode ended (is_success={info['is_success']}); resetting")
            self._obs, _info = self._env.reset()
            self._publish_instruction()


def main():
    rclpy.init()
    node = PushTBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node._env.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
