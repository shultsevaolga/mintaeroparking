import argparse
import enum
import logging
import math
import os
import numpy as np
import pandas as pd
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
        round_minutes: int,
        clusterize: bool,
    ) -> None:
        self.number = int(row["Aircraft_Stand"])
        self.flight_type_on_direction = {
            FlightDirection.ARRIVAL: self._get_flight_type(row["JetBridge_on_Arrival"]),
            FlightDirection.DEPARTURE: self._get_flight_type(row["JetBridge_on_Departure"])
        }

        cluster_fields = [
            row["JetBridge_on_Arrival"],
            row["JetBridge_on_Departure"],
            (int(row["Taxiing_Time"]) // round_minutes) * round_minutes,
        ]

        self.time_to_terminal = {}
        for idx in range(1, 6):
            bus_minutes = row[f"{idx}"]
            cluster_fields.append((bus_minutes // round_minutes) * round_minutes)
            self.time_to_terminal[idx] = timedelta(minutes=bus_minutes)

        self.terminal = int(row["Terminal"]) if not math.isnan(row["Terminal"]) else None
        self.taxiing_time = timedelta(minutes=int(row["Taxiing_Time"]))
        self.taxiing_cost = int(handling_rate.taxiing * as_minutes(self.taxiing_time))

        if self.terminal is None:
            self.stand_type = AircraftStandType.AWAY
            self.stand_rate = handling_rate.away_stand
            cluster_fields.append(-1 if clusterize else self.number)
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
        round_minutes: int=1,
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
        self.stand_time += timedelta(minutes=round_minutes-1)

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


def calc_neighbour_jet_bridge_stand_indices(aircraft_stands: List[object]) -> List[Tuple[int]]:
    previous_stand = None
    previous_stand_idx = None
    neighbour_jet_bridge_stand_indices = []
    jet_bridge_stand_indices = [
        idx for idx, aircraft_stand in enumerate(aircraft_stands)
        if aircraft_stand.stand_type is AircraftStandType.JET_BRIDGE
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
    return neighbour_jet_bridge_stand_indices


def optimize(
    aircraft_stands: List[AircraftStandCluster],
    flights: List[Flight],
    handling_rate: HandlingRatePerMinute,
    round_minutes: int,
    max_seconds: float,
) -> List[AircraftStand]:
    flight_count = len(flights)
    aircraft_stand_count = len(aircraft_stands)

    model = Model()
    model_variables = [[None for j in range(aircraft_stand_count)] for i in range(flight_count)]
    flight_stands = [[None for j in range(aircraft_stand_count)] for i in range(flight_count)]

    inequality_count = 0
    for i, flight in enumerate(flights):
        for j, aircraft_stand in enumerate(aircraft_stands):
            model_variables[i][j] = model.add_var(var_type=BINARY)
            flight_stands[i][j] = FlightStand(
                flight,
                aircraft_stand,
                handling_rate,
                round_minutes,
            )

    min_time = flight_stands[0][0].stand_start
    max_time = flight_stands[0][0].stand_end
    for i, flight in enumerate(flights):
        for j, aircraft_stand in enumerate(aircraft_stands):
            min_time = min(min_time, flight_stands[i][j].stand_start)
            max_time = max(max_time, flight_stands[i][j].stand_end)

    logger.info("Min time: %s, max time: %s", min_time, max_time)
    logger.info("Preparing model")

    neighbour_jet_bridge_stand_indices = calc_neighbour_jet_bridge_stand_indices(aircraft_stands)
    wide_body_flight_indices = [
        idx for idx, flight in enumerate(flights)
        if flight.aircraft_type == AircraftType.WIDE_BODY
    ]

    # each flight has exactly one stand
    for i in range(flight_count):
        model += (xsum(model_variables[i][j] for j in range(aircraft_stand_count)) == 1)

    for current_time in time_range(min_time, max_time, timedelta(minutes=round_minutes)):
        # no more than one flight on each stand at current time
        for j, aircraft_stand in enumerate(aircraft_stands):
            current_time_flight_indices = []
            for i in range(flight_count):
                if flight_stands[i][j].staying_at(current_time):
                    current_time_flight_indices.append(i)

            if current_time_flight_indices:
                inequality_count += 1
                model += (xsum(model_variables[i][j] for i in current_time_flight_indices) <= aircraft_stand.capacity)

        # no more than one wide body flight on near jet bridge stands at current time
        for j1, j2 in neighbour_jet_bridge_stand_indices:
            left_wide_body_indices = []
            right_wide_body_indices = []
            for i in wide_body_flight_indices:
                if flight_stands[i][j1].staying_at(current_time):
                    left_wide_body_indices.append(i)
                if flight_stands[i][j2].staying_at(current_time):
                    right_wide_body_indices.append(i)
            if left_wide_body_indices and right_wide_body_indices:
                inequality_count += 1
                model += (
                    xsum(model_variables[i][j1] for i in left_wide_body_indices) +
                    xsum(model_variables[i][j2] for i in right_wide_body_indices) <= 1
                )

    # optimization objective
    model.objective = minimize(xsum(
        model_variables[i][j] * flight_stands[i][j].cost
        for i in range(flight_count)
        for j in range(aircraft_stand_count)
    ))

    logger.info(
        "Optimizing model, variables: %d, restrictions: %d, inequalities: %d",
        model.num_cols, model.num_rows, inequality_count,
    )

    model.optimize(max_seconds=max_seconds)
    min_cost = int(model.objective_value)
    logger.info("Minimum cost: %d", min_cost)

    result = []
    for i in range(flight_count):
        for j in range(aircraft_stand_count):
            if model_variables[i][j].x > 0.99:
                result.append(flight_stands[i][j])
    return result


def declusterize(flight_stands: List[FlightStand]) -> List[int]:
    flight_stand_numbers = [None] * len(flight_stands)
    aircraft_stands_busy_until = {}
    for flight_stand in sorted(flight_stands, key=lambda item: item.stand_start):
        for number in flight_stand.aircraft_stand.numbers:
            if number not in aircraft_stands_busy_until or aircraft_stands_busy_until[number] <= flight_stand.stand_start:
                aircraft_stands_busy_until[number] = flight_stand.stand_end
                flight_stand_numbers[flight_stand.flight.idx] = number
                break
        else:
            raise RuntimeError("All cluster stands are busy")
    return flight_stand_numbers


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
    parser.add_argument("--output", default="../data/result.csv")
    parser.add_argument("--hours", default=None, type=int)
    parser.add_argument("--round-minutes", default=1, type=int)
    parser.add_argument("--clusterize", action="store_true")
    parser.add_argument("--max-seconds", default=math.inf, type=float)
    params = parser.parse_args()
    print(textwrap.dedent(R"""
        Choose mode:
        [1] Run full optimization
        [2] Round taxiing time to 5-minutes
        [3] Clusterize away stands by rounding down taxiing time to 5-minutes
    """))
    answer = int(input().strip())
    if answer == 1:
        pass
    elif answer == 2:
        params.round_minutes = 5
    elif answer == 3:
        params.round_minutes = 5
        params.clusterize = True
    else:
        raise RuntimeError(f"Unexpected mode: {answer}")

    logger.info("Using mode: %d", answer)

    print("Specify timetable hours limit:")
    params.hours = int(input().strip())

    print("Specify max seconds:")
    params.max_seconds = int(input().strip())
    logger.info("Max optimizing seconds: %d", params.max_seconds)

    return params


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
        AircraftStand(row, handling_rate, params.round_minutes, params.clusterize)
        for _, row in aircraft_stands_data.iterrows()
    ]

    timetable_data = pd.read_csv(os.path.join(params.data_dir, "Timetable_private.csv"), sep=",")
    flights = [
        Flight(idx, row, aircraft_class_by_seats, handling_times_by_aircraft_type)
        for idx, row in timetable_data.iterrows()
    ]

    limit = None
    flights_dt = sorted(flights, key=lambda item: item.dt)
    for i, flight in enumerate(flights_dt):
        flight.idx = i
        if limit is None and flight.dt.hour > params.hours:
            limit = i
    flights_dt = flights_dt[:limit]

    aircraft_stand_clusters = defaultdict(AircraftStandCluster)
    for aircraft_stand in aircraft_stands:
        aircraft_stand_clusters[aircraft_stand.cluster_fields].add(aircraft_stand)
    logger.info("Aircraft stand cluster count: %d", len(aircraft_stand_clusters))

    flight_stands = optimize(
        list(aircraft_stand_clusters.values()),
        flights_dt,
        handling_rate,
        params.round_minutes,
        params.max_seconds,
    )
    flight_stand_numbers = declusterize(flight_stands)
    optimized_cost = get_optmized_cost(aircraft_stands, flights_dt, flight_stand_numbers, handling_rate)
    logger.info("Optimized cost: %d", optimized_cost)
    logger.info("Saving result to %s", params.output)

    fsn = [
        flight_stand_numbers[flight.idx] if flight.idx < len(flight_stand_numbers) else None
        for flight in flights
    ]
    timetable_data["Aircraft_Stand"] = fsn
    timetable_data.to_csv(params.output, sep=",", index=False)


if __name__ == "__main__":
    main()
