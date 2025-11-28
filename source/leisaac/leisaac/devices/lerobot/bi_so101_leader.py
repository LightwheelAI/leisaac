from ..device_base import Device
from .so101_leader import SO101Leader


class BiSO101Leader(Device):
    def __init__(
        self, env, left_port: str = "/dev/ttyACM0", right_port: str = "/dev/ttyACM1", recalibrate: bool = False
    ):
        super().__init__(env, "bi_so101_leader")

        # use left so101 leader as the main device to store state
        print("Connecting to left_so101_leader...")
        self.left_so101_leader = SO101Leader(env, left_port, recalibrate, "left_so101_leader.json", verbose=False)
        print("Connecting to right_so101_leader...")
        self.right_so101_leader = SO101Leader(env, right_port, recalibrate, "right_so101_leader.json", verbose=False)

        self.left_so101_leader._stop_keyboard_listener()
        self.right_so101_leader._stop_keyboard_listener()

    def __str__(self) -> str:
        """Returns: A string containing the information of bi-so101 leader."""
        msg = "Bi-SO101-Leader device for SE(3) control.\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tMove Bi-SO101-Leader to control Bi-SO101-Follower\n"
        msg += (
            "\tIf SO101-Follower can't synchronize with Bi-SO101-Leader, please add --recalibrate and rerun to"
            " recalibrate Bi-SO101-Leader.\n"
        )
        msg += "\t----------------------------------------------\n"
        return msg

    def reset(self):
        self.left_so101_leader.reset()
        self.right_so101_leader.reset()
        super().reset()

    def get_device_state(self):
        return {
            "left_arm": self.left_so101_leader.get_device_state(),
            "right_arm": self.right_so101_leader.get_device_state(),
        }

    def input2action(self):
        ac_dict = super().input2action()
        ac_dict["motor_limits"] = {
            "left_arm": self.left_so101_leader.motor_limits,
            "right_arm": self.right_so101_leader.motor_limits,
        }
        return ac_dict
