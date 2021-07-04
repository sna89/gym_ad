from dataclasses import dataclass
from enum import Enum

MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]


class Action(Enum):
    south = 0
    north = 1
    east = 2
    west = 3
    pickup = 4
    dropoff = 5


class PassengerLocation(Enum):
    red = 0
    green = 1
    yellow = 2
    blue = 3
    in_taxi = 4


class DestinationLocation(Enum):
    red = 0
    green = 1
    yellow = 2
    blue = 3


@dataclass
class TaxiState:
    taxi_row: int
    taxi_column: int
    passenger_loc: PassengerLocation
    destination: DestinationLocation

    def __eq__(self, other):
        if isinstance(other, TaxiState):
            if other.taxi_row == self.taxi_row and \
                    other.taxi_column == self.taxi_column and \
                    other.passenger_loc.value == self.passenger_loc.value and \
                    other.destination.value == self.destination.value:
                return True
            else:
                return False
        else:
            return False

    def __repr__(self):
        return "TaxiRow: {}, TaxiColumn: {}, {}, {}".format(self.taxi_row,
                                                            self.taxi_column,
                                                            self.passenger_loc,
                                                            self.destination)
