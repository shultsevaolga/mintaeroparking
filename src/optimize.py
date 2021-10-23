import argparse
import enum
import logging
import math
import os
import numpy as np
import pandas as pd

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

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


def flight_stand_match_time(
    stand_start_seconds: np.array,
    stand_end_seconds: np.array,
    current_time: int,
    fidx: int,
    sidx: int,
) -> bool:
    # NB: use <= if overlapping on stand end border is prohibited
    return stand_start_seconds[fidx][sidx] <= current_time < stand_end_seconds[fidx][sidx]


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

    def __init__(self, row: pd.DataFrame, handling_rate: HandlingRatePerMinute) -> None:
        self.number = int(row["Aircraft_Stand"])
        self.flight_type_on_direction = {
            FlightDirection.ARRIVAL: self._get_flight_type(row["JetBridge_on_Arrival"]),
            FlightDirection.DEPARTURE: self._get_flight_type(row["JetBridge_on_Departure"])
        }

        cluster_fields = [
            row["JetBridge_on_Arrival"],
            row["JetBridge_on_Departure"],
            row["Taxiing_Time"],
        ]

        self.time_to_terminal = {}
        for idx in range(1, 6):
            bus_minutes = row[f"{idx}"]
            cluster_fields.append(bus_minutes)
            self.time_to_terminal[idx] = timedelta(minutes=bus_minutes)

        self.terminal = int(row["Terminal"]) if not math.isnan(row["Terminal"]) else None
        self.taxiing_time = timedelta(minutes=int(row["Taxiing_Time"]))
        self.taxiing_cost = int(handling_rate.taxiing * as_minutes(self.taxiing_time))

        if self.terminal is None:
            self.stand_type = AircraftStandType.AWAY
            self.stand_rate = handling_rate.away_stand
            # NB: use self.number to prevent clusterization
            cluster_fields.append(-1)
        else:
            self.stand_type = AircraftStandType.JET_BRIDGE
            self.stand_rate = handling_rate.jet_bridge_stand
            cluster_fields.append(self.number)
        self.cluster_fields = tuple(cluster_fields)

    def _get_flight_type(self, value: str) -> Optional[FlightType]:
        if value == "N":
            return None
        return FlightType(value)

    def __str__(self) -> str:
        return f"Aircraft stand [{self.number}]"


class AircraftStandCluster:

    def __init__(self):
        self._base_aircraft_stand = None
        self.stand_type = None
        self.numbers = []

    def add(self, aircraft_stand: AircraftStand) -> None:
        if self.stand_type is None:
            self._base_aircraft_stand = aircraft_stand
            self.stand_type = aircraft_stand.stand_type
        elif self.stand_type is AircraftStandType.JET_BRIDGE:
            raise RuntimeError("Jet bridge stands cluster cannot cantain multiple stands")
        elif aircraft_stand is AircraftStandType.JET_BRIDGE:
            raise RuntimeError("Away stands cluster cannot cantain jet bridge stands")
        self.numbers.append(aircraft_stand.number)

    @property
    def capacity(self) -> int:
        return len(self.numbers)

    def __getattr__(self, attr: str) -> object:
        if attr == "number" and self.stand_type is AircraftStandType.AWAY:
            raise RuntimeError("Cannot get number for away stands cluster")
        if self._base_aircraft_stand is None:
            raise RuntimeError("Empty aircraft clusters are not supported")
        return getattr(self._base_aircraft_stand, attr)


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
        self.dt = row["flight_datetime"].to_pydatetime()
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


@dataclass
class FlightStand:
    flight: Flight
    aircraft_stand_cluster: AircraftStandCluster
    stand_start_time: datetime
    stand_end_time: datetime


def optimize(
    initial_dt: datetime,
    aircraft_stands: List[AircraftStandCluster],
    flights: List[Flight],
    handling_rate: HandlingRatePerMinute,
) -> List[AircraftStand]:
    flight_count = len(flights)
    aircraft_stand_count = len(aircraft_stands)

    stand_start_seconds = np.zeros((flight_count, aircraft_stand_count), dtype=int)
    stand_end_seconds = np.zeros((flight_count, aircraft_stand_count), dtype=int)
    costs = np.zeros((flight_count, aircraft_stand_count), dtype=int)

    model = Model()
    model_variables = [
        [
            model.add_var(var_type=BINARY)
            for j in range(aircraft_stand_count)
        ]
        for i in range(flight_count)
    ]

    inequality_count = 0
    for i, flight in enumerate(flights):
        for j, aircraft_stand in enumerate(aircraft_stands):
            jet_bridge_is_used = (
                aircraft_stand.stand_type is AircraftStandType.JET_BRIDGE and
                aircraft_stand.terminal == flight.terminal and
                aircraft_stand.flight_type_on_direction[flight.direction] == flight.flight_type
            )

            if jet_bridge_is_used:
                stand_time = flight.handling_times_by_stand_type[AircraftStandType.JET_BRIDGE]
            else:
                stand_time = flight.handling_times_by_stand_type[AircraftStandType.AWAY]

            # # NB: Use this retriction if international (domestic) flights
            # #     can NOT use domestic (iternational) jet brdiges with buses
            # if (
            #     aircraft_stand.stand_type is AircraftStandType.JET_BRIDGE and
            #     aircraft_stand.flight_type_on_direction[flight.direction] != flight.flight_type
            # ):
            #     model += (model_variables[i][j] == 0)

            if flight.direction is FlightDirection.ARRIVAL:
                stand_start = flight.dt + aircraft_stand.taxiing_time
                stand_end = stand_start + stand_time
            else:
                stand_end = flight.dt - aircraft_stand.taxiing_time
                stand_start = stand_end - stand_time

            stand_start_seconds[i][j] = (stand_start - initial_dt).total_seconds()
            stand_end_seconds[i][j] = (stand_end - initial_dt).total_seconds()
            costs[i][j] = aircraft_stand.taxiing_cost + aircraft_stand.stand_rate * as_minutes(stand_time)
            if not jet_bridge_is_used:
                bus_time = aircraft_stand.time_to_terminal[flight.terminal]
                costs[i][j] += flight.bus_count * as_minutes(bus_time) * handling_rate.bus

    min_seconds = np.min(stand_start_seconds)
    max_seconds = np.max(stand_end_seconds)

    logger.info("Preparing model")

    previous_stand = None
    previous_stand_idx = None
    neighbour_jet_bridge_stand_indices = []
    jet_bridge_stand_indices = [
        idx for idx in range(aircraft_stand_count)
        if aircraft_stands[idx].stand_type is AircraftStandType.JET_BRIDGE
    ]
    for idx in sorted(jet_bridge_stand_indices, key=lambda idx: aircraft_stands[idx].number):
        aircraft_stand = aircraft_stands[idx]
        if (
            previous_stand is not None and
            aircraft_stand.number == previous_stand.number + 1 and
            aircraft_stand.terminal == previous_stand.terminal
        ):
            neighbour_jet_bridge_stand_indices.append((previous_stand_idx, idx))
        previous_stand = aircraft_stand
        previous_stand_idx = idx

    wide_body_flight_indices = [
        idx for idx, flight in enumerate(flights)
        if flight.aircraft_type == AircraftType.WIDE_BODY
    ]

    # each flight has exactly one stand
    for i in range(flight_count):
        model += (xsum(model_variables[i][j] for j in range(aircraft_stand_count)) == 1)

    for current_time in range(min_seconds, max_seconds+1, int(timedelta(minutes=1).total_seconds())):
        # no more than one flight on each stand at each 5-minutes
        for j, aircraft_stand in enumerate(aircraft_stands):
            current_time_flight_indices = []
            for i in range(flight_count):
                if flight_stand_match_time(stand_start_seconds, stand_end_seconds, current_time, i, j):
                    current_time_flight_indices.append(i)

            if current_time_flight_indices:
                inequality_count += 1
                model += (xsum(model_variables[i][j] for i in current_time_flight_indices) <= aircraft_stand.capacity)

        # no more than one wide body flight on near jet bridge stands at each 5-minutes
        for j1, j2 in neighbour_jet_bridge_stand_indices:
            left_wide_body_indices = []
            right_wide_body_indices = []
            for i in wide_body_flight_indices:
                if flight_stand_match_time(stand_start_seconds, stand_end_seconds, current_time, i, j1):
                    left_wide_body_indices.append(i)
                if flight_stand_match_time(stand_start_seconds, stand_end_seconds, current_time, i, j2):
                    right_wide_body_indices.append(i)
            if left_wide_body_indices and right_wide_body_indices:
                inequality_count += 1
                model += (
                    xsum(model_variables[i][j1] for i in left_wide_body_indices) +
                    xsum(model_variables[i][j2] for i in right_wide_body_indices) <= 1
                )

    # optimization objective
    model.objective = minimize(xsum(
        model_variables[i][j] * costs[i][j]
        for i in range(flight_count)
        for j in range(aircraft_stand_count)
    ))

    logger.info(
        "Optimizing model, variables: %d, restrictions: %d, inequalities: %d",
        model.num_cols, model.num_rows, inequality_count,
    )

    # NB: tune max_seconds for faster optimization
    model.optimize()

    min_cost = int(model.objective_value)
    logger.info("Minimum cost: %d", min_cost)

    flight_stands = []
    for i, flight in enumerate(flights):
        for j, aircraft_stand in enumerate(aircraft_stands):
            if model_variables[i][j].x > 0.99:
                flight_stands.append(FlightStand(
                    flight,
                    aircraft_stand,
                    initial_dt + timedelta(seconds=int(stand_start_seconds[i][j])),
                    initial_dt + timedelta(seconds=int(stand_end_seconds[i][j])),
                ))
    return flight_stands


def declusterize(flight_stands: List[FlightStand]) -> List[int]:
    flight_stand_numbers = [None] * len(flight_stands)
    aircraft_stands_busy_until = {}
    for flight_stand in sorted(flight_stands, key=lambda item: item.stand_start_time):
        for number in flight_stand.aircraft_stand_cluster.numbers:
            # NB: use < if overlapping on stand end border is prohibited
            if number not in aircraft_stands_busy_until or aircraft_stands_busy_until[number] <= flight_stand.stand_start_time:
                aircraft_stands_busy_until[number] = flight_stand.stand_end_time
                flight_stand_numbers[flight_stand.flight.idx] = number
                break
        else:
            raise RuntimeError("All cluster stands are busy")
    return flight_stand_numbers


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--initial-date", default="2019-05-17", type=lambda dt: datetime.strptime(dt, '%Y-%m-%d'))
    parser.add_argument("--output", default="../data/result.csv")
    return parser.parse_args()


def main() -> None:
    params = parse_arguments()
    logger.info("Loading data")

    handling_times_data = pd.read_csv(
        os.path.join(params.data_dir, "Handling_Time_Private.csv"),
        sep=",", index_col="Aircraft_Class",
    )

    handling_times_by_aircraft_type = {
        AircraftType(idx): {
            AircraftStandType.AWAY: timedelta(minutes=int(row["Away_Handling_Time"])),
            AircraftStandType.JET_BRIDGE: timedelta(minutes=int(row["JetBridge_Handling_Time"])),
        } for idx, row in handling_times_data.iterrows()
    }

    handling_rates_data = pd.read_csv(
        os.path.join(params.data_dir, "Handling_Rates_Private.csv"),
        sep=",", index_col="Name",
    )["Value"]

    handling_rate = HandlingRatePerMinute(
        bus=int(handling_rates_data["Bus_Cost_per_Minute"]),
        away_stand=int(handling_rates_data["Away_Aircraft_Stand_Cost_per_Minute"]),
        jet_bridge_stand=int(handling_rates_data["JetBridge_Aircraft_Stand_Cost_per_Minute"]),
        taxiing=int(handling_rates_data["Aircraft_Taxiing_Cost_per_Minute"]),
    )

    aircraft_classes_data = pd.read_csv(
        os.path.join(params.data_dir, "Aircraft_Classes_Private.csv"),
        sep=",", index_col="Max_Seats"
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

    timetable_data = pd.read_csv(os.path.join(params.data_dir, "Timetable_Private.csv"), sep=",", parse_dates=['flight_datetime'])

    flights = [
        Flight(idx, row, aircraft_class_by_seats, handling_times_by_aircraft_type)
        for idx, row in timetable_data.iterrows()
    ]

    aircraft_stand_clusters = defaultdict(AircraftStandCluster)
    for aircraft_stand in aircraft_stands:
        aircraft_stand_clusters[aircraft_stand.cluster_fields].add(aircraft_stand)
    logger.info("Aircraft stand cluster count: %d", len(aircraft_stand_clusters))

    flight_stands = optimize(params.initial_date, list(aircraft_stand_clusters.values()), flights, handling_rate)
    flight_stand_numbers = declusterize(flight_stands)
    timetable_data["Aircraft_Stand"] = flight_stand_numbers

    logger.info("Saving result")
    timetable_data.to_csv(params.output, sep=",", index=False)


if __name__ == "__main__":
    main()
