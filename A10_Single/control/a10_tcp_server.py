import socket
import threading
import base64
import io
import time
from typing import Any, List, cast

import imageio.v2 as imageio
import numpy as np


class A10TcpServer:
    def __init__(self):
        self.server_sock = None
        self.running = False
        self.accept_thread = None

        self.clients_lock = threading.Lock()
        self.client_socks: List[socket.socket] = []

        self.q_lock = threading.Lock()
        self.target_q: List[float] = []

        self.robot_q_lock = threading.Lock()
        self.robot_q: List[float] = []

        self.wrist_rgb_lock = threading.Lock()
        self.wrist_rgb_jpeg_b64 = ""
        self.wrist_rgb_shape = [0, 0, 0]
        self.wrist_rgb_ts = 0.0

    def start(self, port: int) -> bool:
        self.stop()
        try:
            self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass

            self.server_sock.bind(("0.0.0.0", port))
            self.server_sock.listen(5)

            self.running = True
            self.accept_thread = threading.Thread(target=self.accept_loop, daemon=True)
            self.accept_thread.start()
            return True
        except Exception as e:
            print(f"[A10TcpServer] start failed: {e}")
            self.stop()
            return False

    def stop(self):
        self.running = False

        if self.server_sock is not None:
            try:
                self.server_sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self.server_sock.close()
            except Exception:
                pass
            self.server_sock = None

        with self.clients_lock:
            for sock in self.client_socks:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    sock.close()
                except Exception:
                    pass
            self.client_socks.clear()

        if self.accept_thread is not None and self.accept_thread.is_alive():
            self.accept_thread.join(timeout=1.0)
        self.accept_thread = None

    def accept_loop(self):
        while self.running:
            try:
                server_sock = self.server_sock
                if server_sock is None:
                    break
                client_sock, addr = server_sock.accept()
            except OSError:
                break
            except Exception as e:
                if self.running:
                    print(f"[A10TcpServer] accept error: {e}")
                break

            print(f"[A10TcpServer] New connection accepted: {addr}")
            with self.clients_lock:
                self.client_socks.append(client_sock)

            thread = threading.Thread(target=self.reader_loop, args=(client_sock,), daemon=True)
            thread.start()

    def reader_loop(self, client_sock: socket.socket):
        buffer = ""
        while self.running:
            try:
                data = client_sock.recv(1024)
            except Exception as e:
                if self.running:
                    print(f"[A10TcpServer] recv error: {e}")
                break

            if not data:
                print("[A10TcpServer] Client disconnected.")
                break

            buffer += data.decode("utf-8", errors="ignore")
            while True:
                pos = buffer.find("\n")
                if pos == -1:
                    break
                line = buffer[:pos]
                buffer = buffer[pos + 1 :]
                self.process_line(client_sock, line)

        with self.clients_lock:
            if client_sock in self.client_socks:
                self.client_socks.remove(client_sock)

        try:
            client_sock.close()
        except Exception:
            pass

    def send_set_joints(self, q: List[float]) -> bool:
        # Keep latest robot state from control loop. No active broadcast needed.
        with self.robot_q_lock:
            self.robot_q = list(q)
        return True

    def update_wrist_rgb(self, rgb_image: np.ndarray) -> bool:
        """
        Cache latest wrist camera frame for TCP query.
        Input expects uint8 HWC or compatible numpy-like array.
        """
        try:
            frame = np.asarray(rgb_image)
            if frame.ndim != 3 or frame.shape[-1] < 3 or frame.size == 0:
                return False
            frame = frame[..., :3].astype(np.uint8, copy=False)

            buffer = io.BytesIO()
            imageio.imwrite(buffer, frame, format=cast(Any, "jpg"))
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

            with self.wrist_rgb_lock:
                self.wrist_rgb_jpeg_b64 = encoded
                self.wrist_rgb_shape = [int(frame.shape[0]), int(frame.shape[1]), 3]
                self.wrist_rgb_ts = time.time()
            return True
        except Exception:
            return False

    def send_line(self, line: str) -> bool:
        with self.clients_lock:
            if not self.client_socks:
                return False
            socks = list(self.client_socks)

        payload = line.encode("utf-8")
        any_success = False
        dead_socks = []

        for sock in socks:
            ok = True
            sent_total = 0
            while sent_total < len(payload):
                try:
                    sent = sock.send(payload[sent_total:])
                except Exception:
                    ok = False
                    break
                if sent <= 0:
                    ok = False
                    break
                sent_total += sent

            if ok:
                any_success = True
            else:
                dead_socks.append(sock)

        if dead_socks:
            with self.clients_lock:
                for sock in dead_socks:
                    if sock in self.client_socks:
                        self.client_socks.remove(sock)
                    try:
                        sock.close()
                    except Exception:
                        pass

        return any_success

    def get_target_q(self) -> List[float]:
        with self.q_lock:
            return list(self.target_q)

    def _select_leader_vals(self, current_q: List[float]) -> List[float]:
        # Keep compatibility with real robot protocol when 12+ joints exist.
        if len(current_q) >= 12:
            return [current_q[i + 6] for i in range(6)]
        # IsaacLab single-arm fallback: send first 6 arm joints.
        vals = list(current_q[:6])
        while len(vals) < 6:
            vals.append(0.0)
        return vals

    def _select_follower_vals(self, current_q: List[float]) -> List[float]:
        # Keep compatibility with real robot protocol when q[12] exists.
        if len(current_q) >= 13:
            vals = [current_q[i] for i in range(6)]
            vals.append(current_q[12])
            return vals
        # IsaacLab common shape (6 arm + gripper): use q[0:6] + q[6]
        if len(current_q) >= 7:
            vals = [current_q[i] for i in range(6)]
            vals.append(current_q[6])
            return vals
        # Last fallback: pad/truncate to 7 dims.
        vals = list(current_q[:7])
        while len(vals) < 7:
            vals.append(0.0)
        return vals

    def send_leader_state(self, client_sock: socket.socket):
        with self.robot_q_lock:
            current_q = list(self.robot_q)
        vals = self._select_leader_vals(current_q)
        payload = "{\"q\": [" + ", ".join(str(v) for v in vals) + "]}\n"

        try:
            client_sock.sendall(payload.encode("utf-8"))
        except Exception:
            pass

    def send_follower_state(self, client_sock: socket.socket):
        with self.robot_q_lock:
            current_q = list(self.robot_q)
        vals = self._select_follower_vals(current_q)
        payload = "{\"q\": [" + ", ".join(str(v) for v in vals) + "]}\n"

        try:
            client_sock.sendall(payload.encode("utf-8"))
        except Exception:
            pass

    def send_wrist_rgb(self, client_sock: socket.socket):
        with self.wrist_rgb_lock:
            payload = (
                "{"
                + f"\"image_jpeg_b64\": \"{self.wrist_rgb_jpeg_b64}\", "
                + f"\"shape\": [{self.wrist_rgb_shape[0]}, {self.wrist_rgb_shape[1]}, {self.wrist_rgb_shape[2]}], "
                + f"\"ts\": {self.wrist_rgb_ts}"
                + "}\n"
            )

        try:
            client_sock.sendall(payload.encode("utf-8"))
        except Exception:
            pass

    def process_line(self, client_sock: socket.socket, line: str):
        if "GET_LEADER_STATE" in line:
            self.send_leader_state(client_sock)
            return

        if "GET_FOLLOWER_STATE" in line:
            self.send_follower_state(client_sock)
            return

        if "GET_WRIST_RGB" in line:
            self.send_wrist_rgb(client_sock)
            return

        qpos = line.find("\"q\"")
        if qpos == -1:
            print(f"[A10TcpServer] Received unknown command: {line}")
            return

        lbr = line.find("[", qpos)
        rbr = line.find("]", lbr if lbr != -1 else 0)
        if lbr == -1 or rbr == -1 or rbr <= lbr:
            return

        arr = line[lbr + 1 : rbr]
        parsed: List[float] = []

        idx = 0
        while idx < len(arr):
            while idx < len(arr) and arr[idx].isspace():
                idx += 1
            if idx >= len(arr):
                break

            comma = arr.find(",", idx)
            if comma == -1:
                token = arr[idx:]
                idx = len(arr)
            else:
                token = arr[idx:comma]
                idx = comma + 1

            token = token.strip()
            if not token:
                continue

            try:
                parsed.append(float(token))
            except Exception:
                pass

        if parsed:
            with self.q_lock:
                self.target_q = parsed
