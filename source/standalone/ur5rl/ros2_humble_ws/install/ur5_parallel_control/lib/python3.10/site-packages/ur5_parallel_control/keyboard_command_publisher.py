import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import sys
import termios
import tty
import threading
import select


class TerminalKeyboardJointCommandPublisher(Node):
    def __init__(self):
        super().__init__("terminal_keyboard_joint_command_publisher")
        self.publisher = self.create_publisher(Float64MultiArray, "/joint_cmd", 10)
        self.timer_period = 0.1  # 10 Hz
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.angle_delta = [0.0] * 6  # Initialize deltas for 6 joints
        self.lock = threading.Lock()
        self.running = True

        # Save original terminal settings
        self.orig_settings = termios.tcgetattr(sys.stdin)

        # Start the keyboard listener in a separate thread
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()

    def timer_callback(self):
        with self.lock:
            msg = Float64MultiArray()
            msg.data = self.angle_delta
            self.publisher.publish(msg)

    def keyboard_listener(self):
        tty.setcbreak(sys.stdin.fileno())
        try:
            while self.running:
                dr, dw, de = select.select([sys.stdin], [], [], 0)
                if dr:
                    c = sys.stdin.read(1)
                    with self.lock:
                        self.process_key(c)
        except Exception as e:
            self.get_logger().error(f"Keyboard listener exception: {e}")
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.orig_settings)

    def process_key(self, key):
        # Reset angle delta
        self.angle_delta = [0.0] * 6
        if key == "\x1b":  # Escape character
            next1, next2 = sys.stdin.read(2)
            key = key + next1 + next2
            if key == "\x1b[A":  # Up arrow
                self.angle_delta[2] = 1.0
            elif key == "\x1b[B":  # Down arrow
                self.angle_delta[2] = -1.0
        elif key == "q":
            self.get_logger().info("Exiting keyboard control.")
            self.destroy_node()
            rclpy.shutdown()

    def destroy_node(self):
        self.running = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TerminalKeyboardJointCommandPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, node.orig_settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
