import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from pynput import keyboard


class TerminalKeyboardJointCommandPublisher(Node):
    def __init__(self):
        super().__init__("terminal_keyboard_joint_command_publisher")
        self.publisher = self.create_publisher(Float64MultiArray, "/joint_cmd", 10)
        self.angle_delta = [0.0] * 7  # Initialize deltas for 6 joints + gripper
        self.timer_period = 1 / 150  # Publish at 10 Hz
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.gripperswitch = False

        # Start the keyboard listener
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def timer_callback(self):
        if sum(self.angle_delta[:6]) != 0 or self.gripperswitch:
            msg = Float64MultiArray()
            msg.data = self.angle_delta
            self.publisher.publish(msg)
            self.gripperswitch = False

    def on_press(self, key):
        try:
            char = key.char  # For single character keys
            if char.lower() == "w":
                if self.angle_delta[2] == -1.0:
                    self.angle_delta[2] = 0.0
                else:
                    self.angle_delta[2] = -1.0
                # self.get_logger().info("Up command received ('w').")
            elif char.lower() == "s":
                if self.angle_delta[2] == 1.0:
                    self.angle_delta[2] = 0.0
                else:
                    self.angle_delta[2] = 1.0
                # self.get_logger().info("Down command received ('s').")
            elif char.lower() == "a":
                self.angle_delta[6] = 1.0
                self.gripperswitch = True
                # self.get_logger().info("Gripper set to open state I/O 1 ('a').")
            elif char.lower() == "d":
                self.angle_delta[6] = 0.0
                self.gripperswitch = True
                # self.get_logger().info("Gripper set to close state I/O 0 ('d').")
            elif char.lower() == "e":
                # self.get_logger().info("Exiting keyboard control ('e').")
                self.listener.stop()
                rclpy.shutdown()
        except AttributeError:
            # Handle special keys (e.g., Esc)
            if key == keyboard.Key.esc:
                self.get_logger().info("Exiting keyboard control (Esc).")
                self.listener.stop()
                rclpy.shutdown()

    def destroy_node(self):
        self.listener.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TerminalKeyboardJointCommandPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
