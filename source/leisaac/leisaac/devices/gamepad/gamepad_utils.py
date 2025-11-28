from dataclasses import dataclass

import pygame

XBOX_GAMEPAD_MAPPINGS = {
    "buttons": {"A": 0, "B": 1, "X": 2, "Y": 3, "LB": 4, "RB": 5, "L": 9, "R": 10},
    "axes": {"L_X": 0, "L_Y": 1, "LT": 2, "R_X": 3, "R_Y": 4, "RT": 5},  # (x: left/right, y: forward/backward)
    "hats": {"UP": (0, 1), "DOWN": (0, -1), "LEFT": (-1, 0), "RIGHT": (1, 0)},
}


@dataclass
class ControllerState:
    buttons: list[bool]
    axes: list[float]
    hats: list[tuple[int, int]]


class GamepadController:
    def __init__(self, deadzone=0.5):
        self.deadzone = deadzone
        self.joystick = None
        self.name = None
        self.mappings = None

    def start(self):
        """Start the controller and initialize resources"""
        if not pygame.get_init():
            pygame.init()
        if not pygame.joystick.get_init():
            pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad detected. Please connect a gamepad and try again.")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.name = self.joystick.get_name().lower()
        self.mappings = XBOX_GAMEPAD_MAPPINGS if "xbox" in self.name else None

    def stop(self):
        """Stop the controller and release resources"""
        if pygame.joystick.get_init():
            self.joystick.quit()
            pygame.joystick.quit()
        pygame.quit()

    def update(self):
        """Update the controller state - call this once pre frame"""
        for _ in pygame.event.get():
            pass

    def get_state(self) -> ControllerState:
        """Get the current controller state"""
        return ControllerState(
            buttons=[self.joystick.get_button(i) for i in range(self.joystick.get_numbuttons())],
            axes=[
                self.joystick.get_axis(i) if abs(self.joystick.get_axis(i)) > self.deadzone else 0.0
                for i in range(self.joystick.get_numaxes())
            ],
            hats=[self.joystick.get_hat(i) for i in range(self.joystick.get_numhats())],
        )

    def lookup_controller_state(
        self, controller_state: ControllerState, name: str, reverse: bool = False
    ) -> tuple[bool, float]:
        """
        Lookup the controller state for a given name
            return is_activate, state
            state:
                for button_name, return the button state
                for axis_name, return the sign of the axis value
                for hat_name, return the hat value if equal to the related value
        """
        if name in self.mappings["buttons"]:
            is_activate = True
            state = controller_state.buttons[self.mappings["buttons"][name]]
        elif name in self.mappings["axes"]:
            is_activate = controller_state.axes[self.mappings["axes"][name]] != 0
            state = controller_state.axes[self.mappings["axes"][name]] > 0
            if reverse:
                state = not state
        elif name in self.mappings["hats"]:
            is_activate = controller_state.hats[0] != (0, 0)
            state = controller_state.hats[0] == self.mappings["hats"][name]
        else:
            raise ValueError(f"Unknown controller name: {name}")

        return is_activate, state
