# © 2025, Cargo Robotics

import os
import cv2
import pickle
import time
import json
import uuid
import numpy as np
import requests
from py_cargo.packing_algorithm import newPacker
from py_cargo.utils import load_config
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import Int32, String
from rclpy.action import ActionServer
from interfaces.action import Retrieve
from interfaces.srv import AddPackage, StartEnd, CheckOccupied, RemovePackage, NextPackageId, Goal, GetPackageInfo
from interfaces.msg import Point, UpdateAddress, SerialStatus
from py_cargo.reporting import Telemetry
from sqlalchemy import create_engine, Column, Integer, String as SqlString, Float, Boolean, select, delete, text, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from py_cargo.vision_utils import safe_viz_image_writer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

Base = declarative_base()

LOADING_PLATFORM_BUFFER_MM = 50
FORCE_CHUTE_HEIGHT_MM = 20  # packages below this height will always go to the chute


class Package(Base):
    __tablename__ = "packages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(SqlString(36), unique=True, nullable=False)
    address = Column(SqlString)
    latitude = Column(Float)
    longitude = Column(Float)
    height_mm = Column(Integer)
    width_mm = Column(Integer)
    length_mm = Column(Integer)
    weight_g = Column(Integer)
    shelf = Column(Integer)
    x_shelf_mm = Column(Integer)
    y_shelf_mm = Column(Integer)
    dropoff_location = Column(SqlString)
    label_image_paths = Column(SqlString)
    address_verified = Column(Boolean)
    vehicle_id = Column(SqlString)
    sag_height_mm = Column(Integer, default=0)
    failed = Column(Boolean, default=False)

    def to_dict(self):
        return {
            "id": self.id,
            "uuid": self.uuid,
            "address": self.address,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "height_mm": self.height_mm,
            "width_mm": self.width_mm,
            "length_mm": self.length_mm,
            "weight_g": self.weight_g,
            "shelf": self.shelf,
            "x_shelf_mm": self.x_shelf_mm,
            "y_shelf_mm": self.y_shelf_mm,
            "dropoff_location": self.dropoff_location,
            "label_image_paths": self.label_image_paths,
            "address_verified": self.address_verified,
            "vehicle_id": self.vehicle_id,
            "sag_height_mm": self.sag_height_mm,
            "failed": self.failed,
        }


class PackingPlanner(Node):
    def __init__(self):
        super().__init__("packing_planner")
        self.init_logger = Telemetry(ros_node_logger=self.get_logger())
        self.init_logger.start_event("Packing Planner Init")

        self.bridge = CvBridge()

        # Create timer for visualization images
        os.makedirs("/var/cargo/viz", exist_ok=True)
        self.occupancies = None
        self.occupancy_timer = self.create_timer(1.0, self.save_occupancy)

        self.add_package_srv = self.create_service(
            AddPackage, "/packing_planner/add_package", self.add_package_callback
        )

        self.get_package_info_srv = self.create_service(
            GetPackageInfo, "/packing_planner/get_package_info", self.get_package_info_callback
        )

        self.nonconveyable_add_package_srv = self.create_service(
            AddPackage, "/packing_planner/add_nonconveyable_package", self.add_nonconveyable_package_callback
        )

        self.get_next_package_id_srv = self.create_service(
            NextPackageId, "/packing_planner/get_next_package_id", self.get_next_package_id_callback
        )

        self.occupied_loading_area_client = self.create_client(CheckOccupied, "/vision/check_occupied")

        self.mark_failed_package_srv = self.create_service(
            RemovePackage, "/packing_planner/mark_failed_package", self.mark_failed_package_callback
        )

        self.address_update_sub = self.create_subscription(
            UpdateAddress, "/packing_planner/update_address", self.update_address_callback, 10
        )

        self.upload_package_sub = self.create_subscription(
            Int32, "/packing_planner/upload_package", self.upload_package_callback, 10
        )

        self.reset_subscription = self.create_subscription(
            String, "/packing_planner/reset_packages", self.reset_callback, 10
        )

        self.queue_size_sub = self.create_subscription(
            Int32, "/motion_planner/queue_size", self.queue_size_callback, 10
        )

        self.serial_status_sub = self.create_subscription(
            SerialStatus,
            "/serial_comms/serial_status",
            self.serial_status_callback,
            10,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Create a timer for checking cloud addresses every 10 seconds
        self.cloud_address_timer = self.create_timer(10.0, self.check_cloud_addresses)

        self.retrieve_action_server = ActionServer(
            self,
            Retrieve,
            "/packing_planner/retrieve",
            self.retrieve_packages_for_address_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        self.nav_pick_and_place_client = self.create_client(StartEnd, "/motion_planner/pick_and_place")

        self.queue_size = 0
        self.vehicle_id = os.environ.get("VEHICLE_ID", "unknown")
        self.box_in_loading_area = False

        # SQLAlchemy database setup
        self.db_name = "sqlite:////var/cargo/packages.db"
        self.engine = create_engine(self.db_name)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        # TODO ensure that there are no concurrent writes to the database

        self.system_config = load_config()
        if self.system_config is None:
            self.init_logger.error("Configuration failed to load. Aborting initialization.")
            self.init_logger.end_event("Packing Planner Init", success=False)
            return
        self.shelf_heights = [
            self.system_config.bottom_shelf.offset.z,
            self.system_config.middle_shelf_main.offset.z + self.system_config.middle_shelf_main.size.h,
            self.system_config.top_shelf_main.offset.z + self.system_config.top_shelf_main.size.h,
        ]

        top_shelf_size = (
            self.system_config.top_shelf_main.size.l + self.system_config.top_shelf_panhandle.size.l,
            self.system_config.top_shelf_main.size.w,
        )
        middle_shelf_size = (
            self.system_config.middle_shelf_main.size.l + self.system_config.middle_shelf_panhandle.size.l,
            self.system_config.middle_shelf_main.size.w,
        )
        bottom_shelf_size = (self.system_config.bottom_shelf.size.l, self.system_config.bottom_shelf.size.w)

        self.bin_sizes = [bottom_shelf_size, middle_shelf_size, top_shelf_size]
        self.bin_offsets_x = [
            self.system_config.bottom_shelf.offset.x,
            self.system_config.middle_shelf_panhandle.offset.x,
            self.system_config.top_shelf_panhandle.offset.x,
        ]
        self.bin_offsets_y = [
            self.system_config.bottom_shelf.offset.y,
            self.system_config.middle_shelf_main.offset.y,
            self.system_config.top_shelf_main.offset.y,
        ]
        self.init_logger.info(f"Bin sizes: {self.bin_sizes}")
        self.num_shelves = 3
        self.stored_packer = "/var/cargo/packers.pkl"

        self.occupancies = []
        for i in range(self.num_shelves):
            occupancy = np.zeros((self.bin_sizes[i][1], self.bin_sizes[i][0], 3), np.uint8)
            self.occupancies.append(occupancy)

        self.init_packer()

        self.stacked_packages_height = 0

        # draw existing packages from DB
        with self.Session() as session:
            existing_packages = session.execute(select(Package)).scalars().all()
            for package in existing_packages:
                if package.shelf == -1:  # ignore nonconveyable packages
                    continue

                x_shelf_mm = package.x_shelf_mm - self.bin_offsets_x[package.shelf]
                y_shelf_mm = package.y_shelf_mm - self.bin_offsets_y[package.shelf]
                x1 = x_shelf_mm - package.length_mm // 2
                y1 = y_shelf_mm - package.width_mm // 2
                x2 = x_shelf_mm + package.length_mm // 2
                y2 = y_shelf_mm + package.width_mm // 2

                color = (max(50, min(256, package.height_mm)), 0, 0)
                # invert y for display purposes
                cv2.rectangle(
                    self.occupancies[package.shelf],
                    (x1, self.bin_sizes[package.shelf][1] - y1),
                    (x2, self.bin_sizes[package.shelf][1] - y2),
                    color,
                    -1,
                )

                cv2.putText(
                    self.occupancies[package.shelf],
                    str(package.id),
                    ((x1 + x2) // 2, self.bin_sizes[package.shelf][1] - ((y1 + y2) // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

        self.init_logger.end_event("Packing Planner Init", success=True)
        # self.draw_max_rects()
        self.run_benchmark = False  # If set to true, keeps track of failures and resets, runs for a certain number of trials and prints data afterwards, erroring out
        if self.run_benchmark:
            # TESTING: intitialize stuff
            self.fail_counts = [0, 0, 0]  # [shelf0, shelf1, shelf2]
            self.NUM_TRIALS = 5
            self.trial_num = 1
            self.test_results = []  # each trial: {'density:' num, 'num_packages': [shelf0, shelf1, shelf2]}
            self.num_packages = [0, 0, 0]  # number of packages successfully added to each shelf

    def save_occupancy(self):
        """Save the current shelf occupancy state."""
        if self.occupancies is None:
            return
        safe_viz_image_writer("/var/cargo/viz/occupancy_bottom_shelf", self.occupancies[0])
        safe_viz_image_writer("/var/cargo/viz/occupancy_middle_shelf", self.occupancies[1])
        safe_viz_image_writer("/var/cargo/viz/occupancy_top_shelf", self.occupancies[2])

    def queue_size_callback(self, msg):
        self.queue_size = msg.data

    def init_packer(self):
        effector_reach = abs(self.system_config.end_effector.offset.z - self.system_config.y_arm.offset.z)
        self.init_logger.info(f"Effector reach (mm): {effector_reach}")
        if os.path.exists(self.stored_packer):
            self.init_logger.info("Loading stored packing planner")
            with open(self.stored_packer, "rb") as inp:
                self.packers = pickle.load(inp)
                for i in range(len(self.packers)):
                    width = self.packers[i][0].width
                    height = self.packers[i][0].height
                    self.get_logger().info(f"packer {i}: width: {width}, height: {height}")
        else:
            self.init_logger.info("No stored packing planner found, creating new one")
            self.packers = []
            for i in range(self.num_shelves):
                packer = newPacker(
                    rotation=False, max_thickness_diff=effector_reach, gantry_width=self.system_config.y_arm.size.l
                )
                packer.add_bin(*self.bin_sizes[i])
                self.packers.append(packer)

                # add a fake package to each shelf so the cutaway is not used
                if i == 0:
                    # x, y, id
                    length = (
                        self.system_config.loading_platform.size.l
                        + self.system_config.loading_platform.offset.x
                        + LOADING_PLATFORM_BUFFER_MM
                        - self.bin_offsets_x[i]
                    )
                    width = (
                        self.system_config.loading_platform.size.w
                        + self.system_config.loading_platform.offset.y
                        + LOADING_PLATFORM_BUFFER_MM
                        - self.bin_offsets_y[i]
                    )
                    # prevent tall boxes from being packed next to loading zone
                    r = (length, width, -i, self.system_config.loading_platform.size.h)
                elif i == 1:
                    length = (
                        self.system_config.middle_shelf_main.offset.x
                        - self.system_config.middle_shelf_panhandle.offset.x
                    )
                    width = self.system_config.middle_shelf_main.size.w - 1  # prevent packing on panhandle
                    r = (length, width, -i, 0)
                else:  # i == 2
                    length = (
                        self.system_config.top_shelf_main.offset.x - self.system_config.top_shelf_panhandle.offset.x
                    )
                    width = self.system_config.top_shelf_main.size.w - 1  # prevent packing on panhandle
                    r = (length, width, -i, 0)
                packer.add_rect(*r)

                # draw rectangle on occupancy image
                placement = None
                # Find the rectangle we just added
                for rect in self.packers[i].rect_list():
                    if rect[5] == -i:
                        placement = rect

                if placement is None:
                    self.get_logger().error(f"Unable to add fake package to shelf {i}")

                # x, w is length , y and h is width
                b, x, y, l, w, rid = placement
                x_actual = x
                y_actual = y
                length_actual = l
                width_actual = w

                # invert y for display purposes
                color = (50, 50, 50)
                x1 = x_actual
                y1 = self.bin_sizes[i][1] - y_actual
                x2 = x_actual + length_actual
                y2 = self.bin_sizes[i][1] - (y_actual + width_actual)
                cv2.rectangle(self.occupancies[i], (x1, y1), (x2, y2), color, -1)

    def reset_stored_packages(self):
        with self.Session() as session:
            session.execute(delete(Package))
            session.commit()

        if os.path.exists(self.stored_packer):
            os.remove(self.stored_packer)

        self.occupancies = []
        for i in range(self.num_shelves):
            occupancy = np.zeros((self.bin_sizes[i][1], self.bin_sizes[i][0], 3), np.uint8)
            self.occupancies.append(occupancy)

        self.init_packer()

    def reset_callback(self, msg):
        self.get_logger().info("Resetting stored packages")
        self.reset_stored_packages()

    def get_package_info_callback(self, request, response):
        self.get_logger().info(f"Getting package info for package {request.id}")
        with self.Session() as session:
            packages = session.execute(select(Package).where(Package.id.in_([request.id]))).scalars().all()
            packages = [p.to_dict() for p in packages]  # Convert to dict for easier handling
            package = packages[0]
        current_location, _, _ = self.get_package_waypoints(package)
        response.storage_location = current_location
        return response

    def add_nonconveyable_package_callback(self, request, response):
        try:
            # Initialize all required fields for the packages table
            new_package = Package(
                uuid=str(uuid.uuid4()),  # Generate a unique UUID
                address=request.package.address,
                latitude=request.package.latitude,
                longitude=request.package.longitude,
                dropoff_location=request.package.dropoff_location,
                height_mm=0,
                width_mm=0,
                length_mm=0,
                weight_g=0,
                shelf=-1,  # (-1 indicates nonconveyable)
                x_shelf_mm=0,
                y_shelf_mm=0,
                label_image_paths=None,
                address_verified=False,
                vehicle_id=self.vehicle_id,
                sag_height_mm=0,
                failed=False,
            )

            with self.Session() as session:
                session.add(new_package)
                session.commit()
                new_id = new_package.id

            self.get_logger().info("Stored nonconveyable package to the database")
            self.get_logger().info(f"New nonconveyable package ID: {new_id}")
            response.id = new_id
            return response
        except Exception as e:
            self.get_logger().error(f"Failed to add nonconveyable package: {e}")
            response.id = -1
            return response

    # New Algorithm Stuff (temporarily adds a few extra maxrects into the solver)
    def add_rectangle_if_failed(self, r, shelf, new_id):
        """
        Attempts to add a rectangle to a shelf, while temporarily blocking off regions
        that conflict in height with existing rectangles beyond a maximum thickness difference.

        This function is used to handle placement edge cases where certain areas must be
        avoided due to vertical height constraints. It temporarily inserts dummy rectangles
        in regions where adding the new rectangle would violate the maximum allowed height
        difference with neighboring rectangles in order to generate new options for the maxrects.

        Args:
            r (tuple): A tuple representing the rectangle to place, in the format
                    (length, width, rid, height_mm).
            shelf (int): Index of the shelf (or bin) to place the rectangle into.
            new_id (str): Unique identifier for the rectangle being added.

        Returns:
            tuple or None: The successfully placed rectangle as
            (x, y, width, height, thickness, rid), or None if placement failed.

        Notes:
            - Coordinates are based on bottom-left origin (x, y).
            - Temporary "dummy" rectangles are inserted to block invalid placement zones
            and removed after the attempt.
        """
        # find all boxes that are too short/too tall
        length, width, rid, height_mm = r
        max_diff = self.packers[shelf][0].max_thickness_diff

        conflicting_rects = []
        for rect in self.packers[shelf][0].rect_list():
            # find any rectangles that have a difference in height greater than max_diff from r
            existing_thickness = rect[4]
            if abs(existing_thickness - height_mm) > max_diff and existing_thickness > 0:
                conflicting_rects.append(rect)

        merged_x_windows = [
            [rect[0], rect[0] + rect[2], rect[1] + rect[3]] for rect in conflicting_rects
        ]  # [x, x+w, y+h]

        # for each window find all the rectangles in that x range
        dummy_ids = []
        dummy_rects = []
        for x_start, x_end, y_start in merged_x_windows:
            if x_end - x_start > 0:  # make sure no stupid intervals
                rect_above = False
                for rect in self.packers[shelf][0].rect_list():
                    rect_x1 = rect[0]
                    rect_x2 = rect[0] + rect[2]
                    rect_y1 = rect[1]
                    if (not (x_end <= rect_x1 or x_start >= rect_x2)) and (rect_y1 >= y_start):  # overlap in x
                        rect_above = True
                # If there's room above the blocking rectangle
                if rect_above is False:
                    y_end = self.bin_sizes[shelf][1]
                    dummy_w = x_end - x_start
                    dummy_h = self.bin_sizes[shelf][1] - y_start
                    if dummy_w > 0 and dummy_h > 0:
                        dummy_id = f"block_{x_start}_{y_start}_{x_end}_{y_end}"
                        dummy_rect = (x_start, y_start, dummy_w, dummy_h, dummy_id, 1)
                        dummy_rects.append(dummy_rect)
                        add_result = (
                            self.packers[shelf]
                            ._open_bins[0]
                            .add_rect_at_position(x_start, y_start, dummy_w, dummy_h, dummy_id, 1)
                        )
                        dummy_ids.append(dummy_id)

        # add the package
        placement = None

        self.packers[shelf].add_rect(*r)

        # Find the rectangle we just added
        for rect in self.packers[shelf].rect_list():
            if rect[5] == new_id:
                placement = rect

        # get rid of the rectangles we added
        for dummy_id in dummy_ids:
            self.packers[shelf].remove_rect(dummy_id)

        return placement

    def add_package_callback(self, request, response):
        logger = Telemetry(ros_node_logger=self.get_logger())
        logger.start_event("Add Package Callback")

        new_package = Package(
            uuid=str(uuid.uuid4()),  # Generate a unique UUID
            address=request.package.address,
            latitude=request.package.latitude,
            longitude=request.package.longitude,
            height_mm=request.package.height_mm,
            width_mm=request.package.width_mm,
            length_mm=request.package.length_mm,
            weight_g=request.package.weight_g,
            shelf=request.package.shelf,
            x_shelf_mm=request.package.x_shelf_mm,
            y_shelf_mm=request.package.y_shelf_mm,
            dropoff_location=request.package.dropoff_location,
            label_image_paths=None,
            address_verified=False,
            vehicle_id=self.vehicle_id,
            sag_height_mm=request.package.sag_height_mm,
            failed=False,
        )

        if new_package.height_mm > self.system_config.max_package_height:
            logger.warn(f"Package height is too high. Measured {new_package.height_mm} mm")
            logger.end_event("Add Package Callback", success=False)
            response.id = -1
            return response

        with self.Session() as session:
            session.add(new_package)
            session.commit()
            new_id = new_package.id

        logger.info("Stored package to the database")
        logger.info(f"New ID: {new_id}")

        # ensure buffer is at least the length and width of the end effector
        package_buffer_x = max(
            self.system_config.package_dimension_buffer,
            (self.system_config.end_effector.size.l - request.package.length_mm) // 2,
        )
        package_buffer_y = max(
            self.system_config.package_dimension_buffer,
            (self.system_config.end_effector.size.w - request.package.width_mm) // 2,
        )

        logger.info(f"Package buffer: {package_buffer_x}, {package_buffer_y}")

        # x, y
        r = (
            request.package.length_mm + package_buffer_x * 2,
            request.package.width_mm + package_buffer_y * 2,
            new_id,
            request.package.height_mm,
        )

        # shelf splitting based on height
        shelf = None
        if (
            request.package.sag_height_mm
            < self.system_config.shelf_2_max_package_height - self.system_config.package_height_buffer
        ):
            shelf = 2
        elif (
            request.package.sag_height_mm
            < self.system_config.shelf_1_max_package_height - self.system_config.package_height_buffer
        ):
            shelf = 1
        else:
            shelf = 0

        placement = None
        while placement is None and shelf >= 0:
            if self.run_benchmark:
                # TESTING: turn off shelf when failure condition reached
                if self.fail_counts[shelf]:
                    shelf -= 1
                    continue

            if shelf == 0 and request.package.height_mm < self.system_config.loading_platform.size.h:
                logger.info("Package is too short to place on bottom shelf.")
                break

            self.packers[shelf].add_rect(*r)

            # Find the rectangle we just added
            for rect in self.packers[shelf].rect_list():
                if rect[5] == new_id:
                    placement = rect

            # Call the new algorithm to try again if we couldn't find a good placement
            if placement is None:
                placement = self.add_rectangle_if_failed(r, shelf, new_id)

            if placement is None:
                logger.info(f"No room on target shelf {shelf}, trying lower shelf")

                if self.run_benchmark:
                    self.fail_counts[shelf] += 1

                shelf -= 1

        if placement is None:
            logger.warn(f"Failed to find placement for package with ID {new_id}")
            logger.end_event("Add Package Callback", success=False)
            with self.Session() as session:
                session.execute(delete(Package).where(Package.id == new_id))
                session.commit()
            response.id = -1

            if self.run_benchmark:
                new_results = {"density": self.get_packing_efficiency(), "num_packages": self.num_packages}
                logger.info(
                    f"Current Packing Density: {new_results['density']}, # Packages Packed: {new_results['num_packages']}"
                )
                if self.fail_counts[0] and self.fail_counts[1] and self.fail_counts[2]:
                    # TESTING: print out/store the metrics collected, each trial: {'density:' num, 'num_packages': [shelf0, shelf1, shelf2]}
                    self.test_results.append(new_results)
                    # TESTING: increment trial_num, print if reached a certain value
                    self.trial_num += 1
                    if self.trial_num == self.NUM_TRIALS + 1:
                        logger.info(f"{self.test_results=}")
                        raise Exception

                    # TESTING: reset if all shelves have failed
                    self.reset_stored_packages()
                    self.num_packages = [0, 0, 0]
                    self.fail_counts = [0, 0, 0]

            return response

        if self.run_benchmark:
            self.num_packages[shelf] += 1
            self.draw_max_rects()

        with open(self.stored_packer, "wb") as outp:
            pickle.dump(self.packers, outp, pickle.HIGHEST_PROTOCOL)

        # x, w is length , y and h is width
        b, x, y, l, w, rid = placement

        x_actual = x + package_buffer_x
        y_actual = y + package_buffer_y
        length_actual = l - package_buffer_x * 2
        width_actual = w - package_buffer_y * 2
        center_x = x_actual + length_actual // 2
        center_y = y_actual + width_actual // 2

        logger.info(f"Found (local) packing position on shelf {shelf}, center (x,y): {center_x}, {center_y}")

        # print(f"Actual X: {x_actual}")
        # print(f"Actual Y: {y_actual}")
        # print(f"Actual Width: {width_actual}")
        # print(f"Actual Length: {length_actual}")
        # print(f"Center X: {center_x}")
        # print(f"Center Y: {center_y}")

        # color = tuple(np.random.randint(25, 256, 3).tolist())
        # invert y for display purposes
        color = (max(50, min(256, request.package.height_mm)), 0, 0)
        x1 = x_actual
        y1 = self.bin_sizes[shelf][1] - y_actual
        x2 = x_actual + length_actual
        y2 = self.bin_sizes[shelf][1] - (y_actual + width_actual)
        cv2.rectangle(self.occupancies[shelf], (x1, y1), (x2, y2), color, -1)

        # count the number of zero pixels
        # total_pixels = 0
        # zero_pixels = 0
        # for j in range(self.num_shelves):
        #     total_pixels += self.occupancies[j].shape[0] * self.occupancies[j].shape[1]
        #     zero_pixels += np.sum(np.all(self.occupancies[j] == [0, 0, 0], axis=2))

        # self.get_logger().debug(f"Total pixels: {total_pixels}, zero pixels: {zero_pixels}")
        # self.get_logger().debug(f"Occupancy: {1 - zero_pixels / total_pixels}")

        # draw id on rectangle
        cv2.putText(
            self.occupancies[shelf],
            str(new_id),
            ((x1 + x2) // 2, (y1 + y2) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # add offsets of the shelf
        center_x += self.bin_offsets_x[shelf]
        center_y += self.bin_offsets_y[shelf]

        # determine where package should be retrieved
        # TODO use weight as well
        dropoff_location = "back"
        if shelf == 0:
            dropoff_location = "back"
        else:
            fits_in_chute = (
                request.package.length_mm
                < self.system_config.chute_opening.size.l - self.system_config.package_dimension_buffer
                and request.package.width_mm
                < self.system_config.chute_opening.size.w - self.system_config.package_dimension_buffer
            )
            flat_package = request.package.height_mm < FORCE_CHUTE_HEIGHT_MM
            if fits_in_chute or flat_package:
                dropoff_location = "front"
            else:
                dropoff_location = "back"

        with self.Session() as session:
            package = session.get(Package, new_id)
            if package:
                package.shelf = shelf
                package.x_shelf_mm = center_x
                package.y_shelf_mm = center_y
                package.dropoff_location = dropoff_location
                session.commit()

        response.id = new_id
        response.shelf = shelf
        response.x_shelf_mm = center_x
        response.y_shelf_mm = center_y
        logger.info(
            f"Found packing position center (x,y): {center_x}, {center_y} for length {length_actual} and width {width_actual} and height {request.package.height_mm}"
        )
        logger.end_event("Add Package Callback", success=True)
        return response

    def get_packing_efficiency(self):
        total_pixels = 0
        blue_pixels = 0

        for j in range(self.num_shelves):
            shelf_img = self.occupancies[j]
            h, w, _ = shelf_img.shape
            total_pixels += h * w

            # Blue pixel condition: blue > 0, green == 0, red == 0
            blue_channel = shelf_img[:, :, 0]
            green_channel = shelf_img[:, :, 1]
            red_channel = shelf_img[:, :, 2]

            blue_mask = (blue_channel > 0) & (green_channel == 0) & (red_channel == 0)
            blue_pixels += np.sum(blue_mask)

        self.get_logger().debug(f"Total pixels: {total_pixels}, blue pixels: {blue_pixels}")
        self.get_logger().debug(f"Occupancy: {blue_pixels / total_pixels:.4f}")

        return blue_pixels / total_pixels

    def draw_max_rects(self):
        for shelf_num in [2, 1, 0]:
            for rect in self.packers[shelf_num]._open_bins[0]._max_rects:
                x1 = rect.x
                y1 = self.bin_sizes[shelf_num][1] - rect.y - rect.height
                x2 = rect.x + rect.width
                y2 = self.bin_sizes[shelf_num][1] - (rect.y)
                cv2.rectangle(self.occupancies[shelf_num], (x1, y1), (x2, y2), (100, 100, 100), 5)
                cv2.circle(self.occupancies[shelf_num], center=(x1, y2), radius=20, color=(0, 255, 0), thickness=-1)

    def draw_package_status_box(self, current_location, package, color, id=None):
        shelf = package["shelf"]
        if shelf < 0:  # Only draw if it's a real shelf
            return
        x_actual = current_location.x - self.bin_offsets_x[shelf] - package["length_mm"] // 2
        y_actual = current_location.y - self.bin_offsets_y[shelf] - package["width_mm"] // 2
        length_actual = package["length_mm"]
        width_actual = package["width_mm"]
        cv2.rectangle(
            self.occupancies[shelf],
            (x_actual, self.bin_sizes[shelf][1] - y_actual),
            (x_actual + length_actual, self.bin_sizes[shelf][1] - (y_actual + width_actual)),
            color,
            -1,
        )
        # draw id on rectangle
        if id is not None:
            cv2.putText(
                self.occupancies[shelf],
                str(id),
                (x_actual + length_actual // 2, self.bin_sizes[shelf][1] - y_actual - width_actual // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

    async def retrieve_packages_for_address_callback(self, goal_handle):
        logger = Telemetry(ros_node_logger=self.get_logger())
        logger.start_event("Retrieve Packages Callback")
        self.stacked_packages_height = 0
        package_ids = goal_handle.request.package_ids

        with self.Session() as session:
            if package_ids == "random":
                logger.info("Retrieving random packages")
                # Use text() for raw SQL expressions
                package = session.execute(
                    select(Package).where(Package.failed == False).order_by(text("RANDOM()")).limit(1)
                ).scalar_one_or_none()
                package_ids = [str(package.id)] if package else []
                if not package_ids:
                    logger.error("Failed to find any packages")
                    logger.end_event("Retrieve Packages Callback", success=False)
                    goal_handle.abort()
                    return Retrieve.Result(success=False, description="Failed to find any packages")
            elif package_ids == "all":
                logger.info("Retrieving all packages")
                package_ids = [
                    str(p) for p in session.execute(select(Package.id).where(Package.failed == False)).scalars().all()
                ]
                if not package_ids:
                    logger.error("Failed to find any packages")
                    logger.end_event("Retrieve Packages Callback", success=False)
                    goal_handle.abort()
                    return Retrieve.Result(success=False, description="Failed to find any packages")
            else:
                # Split the comma-separated string into a list of IDs
                package_ids = [int(id.strip()) for id in package_ids.split(",")]
                logger.info(f"Retrieving packages with IDs: {package_ids}")

            logger.info(f"Retrieving packages with IDs: {package_ids}")
            packages = session.execute(select(Package).where(Package.id.in_(package_ids))).scalars().all()
            packages = [p.to_dict() for p in packages]  # Convert to dict for easier handling

            if not packages:
                description = f"Failed to find packages with IDs: {package_ids}"
                logger.error(description)
                logger.end_event("Retrieve Packages Callback", success=False)
                goal_handle.abort()
                return Retrieve.Result(success=False, description=description)

        # Sort packages by decreasing shelf order, and each shelf by dropoff location (front first)
        packages_by_id = {}
        package_ids = []
        for package in packages:
            packages_by_id[package["id"]] = package
        shelf_groups = {}
        for package in packages:
            shelf = package["shelf"]
            if shelf not in shelf_groups:
                shelf_groups[shelf] = []
            shelf_groups[shelf].append(package)

        # Sort shelves in decreasing order
        for shelf in sorted(shelf_groups.keys(), reverse=True):
            # Sort packages within each shelf by dropoff location (front first)
            front_packages = [p for p in shelf_groups[shelf] if p["dropoff_location"] == "front"]
            back_packages = [p for p in shelf_groups[shelf] if p["dropoff_location"] != "front"]

            # Add front packages first, then back packages
            for package in front_packages + back_packages:
                package_ids.append(package["id"])

        # Start retrieving loop
        failed_packages = package_ids.copy()

        for id in package_ids:
            package = packages_by_id[id]
            current_location, desired_location, dropoff_location = self.get_package_waypoints(package)
            if current_location is None or desired_location is None or dropoff_location == "NC":
                continue
            request = StartEnd.Request()
            request.start_point = current_location
            request.end_point = desired_location
            request.id = id
            request.package_length = package["length_mm"]
            request.package_width = package["width_mm"]
            request.package_height = package["height_mm"]
            logger.info("Current package location: " + str(current_location))
            logger.info("Desired package location: " + str(desired_location))

            # if queue size is > 1, wait for current operation to finish
            if self.queue_size > 0:
                self.get_logger().info(f"Queue size is {self.queue_size}, waiting for current retrieve to finish")
                while self.queue_size > 0:
                    time.sleep(0.1)

            if dropoff_location == "back":
                self.get_logger().info("Back dropoff requested")

                while self.box_in_loading_area:
                    self.get_logger().info("Box detected in loading/unloading area. Please remove to continue...")
                    time.sleep(1)

                self.get_logger().info("Loading area is clear")

            logger.start_timer(f"Pick and Place Call: {id}")
            try:
                # Draw yellow rectangle to show we are working on it
                self.draw_package_status_box(current_location, package, (0, 255, 255), id)  # Yellow

                self.get_logger().info("Calling pick and place")
                response = await self.nav_pick_and_place_client.call_async(request)
                if not response.success:
                    logger.stop_timer(f"Pick and Place Call: {id}")
                    self.draw_package_status_box(current_location, package, (0, 0, 255), id)  # Red
                    if response.last_action == "PICK AND RAISE":
                        continue
                    else:
                        raise Exception(f"Failed to perform action: {response.last_action}")
            except Exception as e:
                logger.error(f"Failed in attempt to pick and place. {e}")
                logger.end_event(f"Retrieve Packages Callback", success=False)
                goal_handle.abort()
                return Retrieve.Result(
                    success=False,
                    description="Failed in attempt to pick and place",
                    extraction_required=True,
                    package_ids=[id],
                )
            logger.stop_timer(f"Pick and Place Call: {id}")
            self.draw_package_status_box(current_location, package, (0, 0, 0))  # Black

            # Remove package from DB and planner
            self.remove_package(package)
            failed_packages.remove(id)

        if failed_packages:
            error_str = f"Retrieve finished with failed packages: {failed_packages}"
            logger.error(error_str)
            logger.end_event("Retrieve Packages Callback", success=False)
            goal_handle.abort()
            return Retrieve.Result(
                success=False,
                description=error_str,
                extraction_required=True,
                package_ids=failed_packages,
            )

        goal_handle.succeed()
        logger.end_event("Retrieve Packages Callback", success=True)
        return Retrieve.Result(success=True)

    def get_package_waypoints(self, package):
        if package["shelf"] == -1 or package["dropoff_location"] == "NC":
            return None, None, "NC"

        current_location = Point()
        current_location.x = package["x_shelf_mm"]
        current_location.y = package["y_shelf_mm"]
        current_location.z = self.shelf_heights[package["shelf"]] + package["height_mm"]

        if package["address"] == "":
            # if no address is provided, it is a pickup or we were unable to find an address. Send to back
            package["dropoff_location"] = "back"

        if package["dropoff_location"] == "front":
            desired_location = Point()
            desired_location.x = self.system_config.chute_opening.offset.x + int(
                0.5 * self.system_config.chute_opening.size.l
            )
            desired_location.y = self.system_config.chute_opening.offset.y + int(
                0.5 * self.system_config.chute_opening.size.w
            )
            # allow some buffer to release above the opening
            fits_in_chute = (
                package["length_mm"]
                < self.system_config.chute_opening.size.l - self.system_config.package_dimension_buffer
                and package["width_mm"]
                < self.system_config.chute_opening.size.w - self.system_config.package_dimension_buffer
            )
            if fits_in_chute:
                desired_location.z = (
                    self.system_config.chute_opening.offset.z
                    + self.system_config.package_height_buffer
                    + package["height_mm"]
                )
            else:
                # let fall into the chute from the shelf height
                desired_location.z = self.shelf_heights[package["shelf"]] + package["height_mm"]
        else:
            # stack in the back
            desired_location = Point()
            desired_location.x = self.system_config.cutaway_x
            desired_location.y = self.system_config.cutaway_y
            # not stacking yet
            desired_location.z = (
                self.system_config.loading_zone_z + package["height_mm"] + self.system_config.drop_height_buffer
            )
            # TODO adjust height higher if there will be a collision with a shelf
            # desired_location.z = (
            #     self.system_config.loading_zone_z + height_mm + self.system_config.drop_height_buffer + self.stacked_packages_height
            # )
            self.stacked_packages_height += package["height_mm"]
            # TODO do we need to handle packages being removed while they are being unloaded?
            # TODO how do we handle too many packages at one address?

        self.get_logger().info(f"Removing from shelf: {package['shelf']}")
        self.get_logger().info(f"Dropoff location: {package['dropoff_location']}")
        return current_location, desired_location, package["dropoff_location"]

    def remove_package(self, package):
        # get data from DB
        self.remove_package_from_db(package["id"])
        self.remove_package_from_planner(
            package["id"],
            package["shelf"],
            package["x_shelf_mm"],
            package["y_shelf_mm"],
            package["length_mm"],
            package["width_mm"],
        )

    def remove_package_from_db(self, id):
        with self.Session() as session:
            package = session.get(Package, id)
            if package is None:
                self.get_logger().error(f"Cannot remove, failed to find package with ID {id}")
            # remove from DB
            session.delete(package)
            session.commit()

            # check the number of packages remaining in the database, reset if back down to 0
            count = session.execute(select(func.count()).select_from(Package).where(Package.failed == False)).scalar()
            if count == 0:
                self.get_logger().info("No valid packages remaining in the database. Resetting planner...")
                self.reset_stored_packages()

    def remove_package_from_planner(self, id, shelf, x_shelf_mm, y_shelf_mm, length_mm, width_mm):
        self.packers[shelf].remove_rect(id)

        # remove from occupancy grid visualization
        # remove shelf offsets
        x_shelf_mm -= self.bin_offsets_x[shelf]
        y_shelf_mm -= self.bin_offsets_y[shelf]

        x_actual = x_shelf_mm - length_mm // 2
        y_actual = y_shelf_mm - width_mm // 2
        length_actual = length_mm
        width_actual = width_mm
        cv2.rectangle(
            self.occupancies[shelf],
            (x_actual, self.bin_sizes[shelf][1] - y_actual),
            (x_actual + length_actual, self.bin_sizes[shelf][1] - (y_actual + width_actual)),
            (0, 0, 0),
            -1,
        )

    def mark_failed_package_callback(self, request, response):
        id = request.id

        shelf = None
        x_shelf_mm = None
        y_shelf_mm = None
        length_mm = None
        width_mm = None

        with self.Session() as session:
            package = session.get(Package, id)
            if package is None:
                self.get_logger().error(f"Cannot mark failed, failed to find package with ID {id}")
                return response
            else:
                # Store the values we need before committing and closing the session
                shelf = package.shelf
                x_shelf_mm = package.x_shelf_mm
                y_shelf_mm = package.y_shelf_mm
                length_mm = package.length_mm
                width_mm = package.width_mm

                package.failed = True
                session.commit()

        self.remove_package_from_planner(id, shelf, x_shelf_mm, y_shelf_mm, length_mm, width_mm)

        return response

    def update_address_callback(self, msg):
        image_paths = json.dumps(msg.image_paths)
        self.get_logger().info(f"Updating address for package ID {msg.id} to {msg.address}")

        with self.Session() as session:
            package = session.get(Package, msg.id)
            if package:
                package.address = msg.address
                package.latitude = msg.latitude
                package.longitude = msg.longitude
                package.label_image_paths = image_paths
                package.address_verified = False
                session.commit()

    def get_next_package_id_callback(self, request, response):
        """Return the next available package ID (max id + 1, or 1 if none exist) as Int32.data."""
        try:
            with self.Session() as session:
                max_id = session.query(func.max(Package.id)).scalar()
                if max_id is None:
                    next_id = 1
                else:
                    next_id = max_id + 1
            response.data = next_id
        except Exception as e:
            response.data = -1
        return response

    def upload_package_callback(self, msg):
        """Upload package data to the Cargo Robotics API"""
        logger = Telemetry(ros_node_logger=self.get_logger())
        logger.start_event("Upload Package to API")

        try:
            # Get package data from database
            with self.Session() as session:
                package = session.get(Package, msg.data)
                if package is None:
                    logger.error(f"Cannot upload, failed to find package with ID {msg.data}")
                    logger.end_event("Upload Package to API", success=False)
                    return

                # Convert package to dictionary
                package_data = package.to_dict()

            # Upload to Cargo Robotics API
            logger.info(f"Uploading package {msg.data} to Cargo Robotics API")
            response = requests.post(
                "https://cargo-robotics.com/insert_package?secret=for_cargo_eyes_only", json=package_data
            )

            if response.status_code == 200:
                logger.info(f"Successfully uploaded package {msg.data} to API. Response: {response.json()}")
                logger.end_event("Upload Package to API", success=True)
            else:
                logger.error(f"Failed to upload package {msg.data} to API. Status code: {response.status_code}")
                logger.end_event("Upload Package to API", success=False)

        except Exception as e:
            logger.error(f"Error uploading package {msg.data} to API: {str(e)}")
            logger.end_event("Upload Package to API", success=False)

    def serial_status_callback(self, msg):
        self.box_in_loading_area = msg.platform_light_curtain
        self.current_arm_position = [msg.current_x, msg.current_y, msg.current_z]

    def check_cloud_addresses(self):
        """Check and update package addresses from the cloud service every 10 seconds."""
        logger = Telemetry(ros_node_logger=self.get_logger())
        logger.start_event("Check Cloud Addresses")

        try:
            # Get all packages that need address verification
            with self.Session() as session:
                packages_to_verify = (
                    session.execute(select(Package).where(Package.address_verified == False)).scalars().all()
                )

                if not packages_to_verify:
                    # No packages need verification
                    logger.info("No packages need address verification")
                    logger.end_event("Check Cloud Addresses", success=True)
                    return

                # Log how many packages need verification
                logger.info(f"Checking cloud addresses for {len(packages_to_verify)} packages")

                # Extract UUIDs from packages that need verification
                uuids_to_verify = [package.uuid for package in packages_to_verify]
                uuids_param = ",".join(uuids_to_verify)

                # Query the server for these packages
                try:
                    logger.info(f"Querying server for {len(uuids_to_verify)} packages")
                    response = requests.get(
                        f"https://cargo-robotics.com/packages?secret=for_cargo_eyes_only&uuids={uuids_param}",
                        timeout=10,
                    )

                    if response.status_code == 200:
                        server_data = response.json()
                        if "packages" in server_data:
                            # Process each address group from the server
                            for address, server_packages in server_data["packages"].items():
                                for server_package in server_packages:
                                    # Find matching package in our local database
                                    for local_package in packages_to_verify:
                                        if local_package.uuid == server_package["uuid"]:
                                            # Check if the server has verified the address
                                            if server_package.get("address_verified", False):
                                                logger.info(
                                                    f"Server has verified address for package {local_package.id}: {address}"
                                                )
                                                # Update local package with server data
                                                local_package.address_verified = True
                                                local_package.address = address
                                                local_package.latitude = server_package.get(
                                                    "latitude", local_package.latitude
                                                )
                                                local_package.longitude = server_package.get(
                                                    "longitude", local_package.longitude
                                                )
                                            break
                    else:
                        logger.warn(f"Server returned status code {response.status_code} when querying packages")
                except Exception as e:
                    logger.error(f"Error querying server for packages: {str(e)}")

                # Commit all changes
                session.commit()
                logger.info("Address verification completed")

        except Exception as e:
            logger.error(f"Error in check_cloud_addresses: {str(e)}")
            logger.end_event("Check Cloud Addresses", success=False)
            return

        logger.end_event("Check Cloud Addresses", success=True)


def main(args=None):
    rclpy.init(args=args)
    packing_planner = PackingPlanner()

    # Use a multithreaded executor
    executor = MultiThreadedExecutor()
    executor.add_node(packing_planner)

    try:
        executor.spin()
    finally:
        packing_planner.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
