import argparse
import enum
import logging
import math
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from mip import minimize, xsum, Model, BINARY


BUS_CAPACITY = 80


def make_logger(log_level: int = logging.DEBUG) -> logging.Logger:
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s [%(filename)s:%(lineno)d]\t%(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(log_level)

    logger = logging.getLogger("svo")
    logger.setLevel(log_level)
    logger.addHandler(handler)
    return logger


logger = make_logger()


def as_minutes(td: timedelta) -> int:
    return int(td.total_seconds() / 60)


def time_range(start: datetime, end: datetime, step: timedelta) -> datetime:
    while start <= end:
        yield start
        start += step


class FlightType(enum.Enum):
    DOMESTIC = "D"
    INTERNATIONAL = "I"


class AircraftType(enum.Enum):
    REGIONAL = "Regional"
    NARROW_BODY = "Narrow_Body"
    WIDE_BODY = "Wide_Body"


class FlightDirection(str, enum.Enum):
    ARRIVAL = "A"
    DEPARTURE = "D"


class AircraftStandType(enum.Enum):
    AWAY = 0
    JET_BRIDGE = 1


@dataclass
class HandlingRatePerMinute:
    bus: int
    away_stand: int
    jet_bridge_stand: int
    taxiing: int


class AircraftStand:

    def __init__(
        self,
        row: pd.DataFrame,
        handling_rate: HandlingRatePerMinute,
    ) -> None:
        self.number = int(row["Aircraft_Stand"])
        self.flight_type_on_direction = {
            FlightDirection.ARRIVAL: self._get_flight_type(row["JetBridge_on_Arrival"]),
            FlightDirection.DEPARTURE: self._get_flight_type(row["JetBridge_on_Departure"])
        }

        self.time_to_terminal = {}
        for idx in range(1, 6):
            bus_minutes = row[f"{idx}"]
            self.time_to_terminal[idx] = timedelta(minutes=bus_minutes)

        self.terminal = int(row["Terminal"]) if not math.isnan(row["Terminal"]) else None
        self.taxiing_time = timedelta(minutes=int(row["Taxiing_Time"]))
        self.taxiing_cost = int(handling_rate.taxiing * as_minutes(self.taxiing_time))

        if self.terminal is None:
            self.stand_type = AircraftStandType.AWAY
            self.stand_rate = handling_rate.away_stand
        else:
            self.stand_type = AircraftStandType.JET_BRIDGE
            self.stand_rate = handling_rate.jet_bridge_stand

    def _get_flight_type(self, value: str) -> Optional[FlightType]:
        if value == "N":
            return None
        return FlightType(value)

    def __str__(self) -> str:
        return f"Aircraft stand [{self.number}]"


class Flight:

    def __init__(
        self,
        idx: int,
        row: pd.DataFrame,
        aircraft_class_by_seats: Dict[int, AircraftType],
        handling_times_by_aircraft_type: Dict[AircraftType, Dict[AircraftStandType, timedelta]],
    ) -> None:
        self.idx = idx
        self.number = int(row["flight_number"])
        self.direction = FlightDirection(row["flight_AD"])
        self.dt = datetime.strptime(row["flight_datetime"], "%d.%m.%Y %H:%M")
        self.flight_type = FlightType(row["flight_ID"])
        self.capacity = int(row["flight_AC_PAX_capacity_total"])
        self.pax_count = int(row["flight_PAX"])
        self.bus_count = math.ceil(self.pax_count / BUS_CAPACITY)
        self.terminal = int(row["flight_terminal_#"])

        self.aircraft_type = None
        for max_seats in sorted(aircraft_class_by_seats):
            if max_seats >= self.capacity:
                self.aircraft_type = aircraft_class_by_seats[max_seats]
                break
        if self.aircraft_type is None:
            raise RuntimeError(f"Flight {self.number} has too big capacity: {self.capacity}")

        self.handling_times_by_stand_type = handling_times_by_aircraft_type[self.aircraft_type]

    def __str__(self):
        return f"Flight [{self.number}]"


class FlightStand:

    def __init__(
        self,
        flight: Flight,
        aircraft_stand: object,
        handling_rate: HandlingRatePerMinute,
    ):
        self.flight = flight
        self.aircraft_stand = aircraft_stand
        self.jet_bridge_is_used = (
            aircraft_stand.stand_type is AircraftStandType.JET_BRIDGE and
            aircraft_stand.terminal == flight.terminal and
            aircraft_stand.flight_type_on_direction[flight.direction] == flight.flight_type
        )

        handling_times = flight.handling_times_by_stand_type
        if self.jet_bridge_is_used:
            self.stand_time = handling_times[AircraftStandType.JET_BRIDGE]
        else:
            self.stand_time = handling_times[AircraftStandType.AWAY]

        if flight.direction is FlightDirection.ARRIVAL:
            self.stand_start = flight.dt + aircraft_stand.taxiing_time
            self.stand_end = self.stand_start + self.stand_time
        else:
            self.stand_end = flight.dt - aircraft_stand.taxiing_time
            self.stand_start = self.stand_end - self.stand_time

        self.cost = aircraft_stand.taxiing_cost + aircraft_stand.stand_rate * as_minutes(self.stand_time)
        if not self.jet_bridge_is_used:
            bus_time = aircraft_stand.time_to_terminal[flight.terminal]
            self.cost += flight.bus_count * as_minutes(bus_time) * handling_rate.bus

    def staying_at(self, dt: datetime) -> bool:
        return self.stand_start <= dt < self.stand_end


def get_optmized_cost(
    aircraft_stands: List[AircraftStand],
    flights: List[Flight],
    flight_stand_numbers: List[int],
    handling_rate: HandlingRatePerMinute,
) -> int:
    aircraft_stand_by_number = {
        aircraft_stand.number: aircraft_stand
        for aircraft_stand in aircraft_stands
    }
    optimized_cost = 0
    for i, flight in enumerate(flights):
        aircraft_stand_number = flight_stand_numbers[i]
        flight_stand = FlightStand(flight, aircraft_stand_by_number[aircraft_stand_number], handling_rate)
        optimized_cost += flight_stand.cost
    return optimized_cost


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--result", default="../data/result.csv")
    return parser.parse_args()


def main() -> None:
    params = parse_arguments()
    logger.info("Loading data")

    handling_times_data = pd.read_csv(
        os.path.join(params.data_dir, "Handling_Time_Public.csv"),
        sep=",", index_col="Aircraft_Class",
    )

    handling_times_by_aircraft_type = {
        AircraftType(idx): {
            AircraftStandType.AWAY: timedelta(minutes=int(row["Away_Handling_Time"])),
            AircraftStandType.JET_BRIDGE: timedelta(minutes=int(row["JetBridge_Handling_Time"])),
        } for idx, row in handling_times_data.iterrows()
    }

    handling_rates_data = pd.read_csv(
        os.path.join(params.data_dir, "Handling_Rates_SVO_Private.csv"),
        sep=",", index_col="Name",
    )["Value"]

    handling_rate = HandlingRatePerMinute(
        bus=int(handling_rates_data["Bus_Cost_per_Minute"]),
        away_stand=int(handling_rates_data["Away_Aircraft_Stand_Cost_per_Minute"]),
        jet_bridge_stand=int(handling_rates_data["JetBridge_Aircraft_Stand_Cost_per_Minute"]),
        taxiing=int(handling_rates_data["Aircraft_Taxiing_Cost_per_Minute"]),
    )

    aircraft_classes_data = pd.read_csv(
        os.path.join(params.data_dir, "AirCraftClasses_Public.csv"),
        sep=",", index_col="Max_Seats",
    )
    aircraft_class_by_seats = {
        int(idx): AircraftType(row["Aircraft_Class"])
        for idx, row in aircraft_classes_data.iterrows()
    }

    aircraft_stands_data = pd.read_csv(os.path.join(params.data_dir, "Aircraft_Stands_Private.csv"), sep=",")
    aircraft_stands = [
        AircraftStand(row, handling_rate)
        for _, row in aircraft_stands_data.iterrows()
    ]

    timetable_data = pd.read_csv(os.path.join(params.data_dir, "result.csv"), sep=",")
    timetable_data.dropna(inplace=True)
    timetable_data.reset_index(drop=True, inplace=True)

    flights = [
        Flight(idx, row, aircraft_class_by_seats, handling_times_by_aircraft_type)
        for idx, row in timetable_data.iterrows()
    ]

    flight_stands_numbers = timetable_data["Aircraft_Stand"]
    cost = get_optmized_cost(aircraft_stands, flights, flight_stands_numbers, handling_rate)
    logger.info("Optimized cost: %d", cost)

    flight_stands = []
    aircraft_stand_by_number = {aircraft_stand.number: aircraft_stand for aircraft_stand in aircraft_stands}
    for flight, number in zip(flights, flight_stands_numbers):
        flight_stands.append(FlightStand(flight, aircraft_stand_by_number[number], handling_rate))

    min_time = flight_stands[0].stand_start
    max_time = flight_stands[0].stand_end
    for flight_stand in flight_stands:
        min_time = min(min_time, flight_stand.stand_start)
        max_time = max(max_time, flight_stand.stand_end)

    colors = ("green", "blue", "darkcyan", "red", "teal")
    terminals = (1, 2, 3, 5, None)
    for terminal, color in zip(terminals, colors):
        times = []
        terminal_stands = [aircraft_stand for aircraft_stand in aircraft_stands if aircraft_stand.terminal == terminal]
        terminal_stands_idx = {aircraft_stand.number: idx for idx, aircraft_stand in enumerate(terminal_stands)}
        timeline = np.zeros((as_minutes(max_time - min_time)+1, len(terminal_stands)))
        for i, current_time in enumerate(time_range(min_time, max_time, timedelta(minutes=1))):
            times.append(f"{current_time:%H:%M}")
            for flight_stand in flight_stands:
                if flight_stand.aircraft_stand.terminal == terminal and flight_stand.staying_at(current_time):
                    timeline[i][terminal_stands_idx[flight_stand.aircraft_stand.number]] += 1
        timeline_df = pd.DataFrame(timeline)
        timeline_df["times"] = times
        timeline_df = timeline_df.set_index("times")
        timeline_df.to_csv(f"../data/timeline_{terminal}.csv")


if __name__ == "__main__":
    main()
