#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
"""
An experimental kinematics parser that reads the robot from a USD file or stage.

Basic loading of simple robots should work but more complex robots may not be supported. E.g.,
mimic joints cannot be parsed correctly. Use the URDF parser (:class:`~UrdfKinematicsParser`)
for more complex robots.
"""

# Standard Library
from typing import Dict, List, Optional, Tuple

# Third Party
import numpy as np

# CuRobo
from curobo.cuda_robot_model.kinematics_parser import KinematicsParser, LinkParams
from curobo.cuda_robot_model.types import JointType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util.logger import log_error
from pxr import Gf

try:
    # Third Party
    from pxr import Usd, UsdPhysics
except ImportError:
    raise ImportError(
        "usd-core failed to import, install with pip install usd-core"
        + " NOTE: Do not install this if using with ISAAC SIM."
    )


class UsdKinematicsParser(KinematicsParser):
    """An experimental kinematics parser from USD.

    Current implementation does not account for link geometry transformations after a joints.
    Also, cannot read mimic joints.

    """

    def __init__(
        self,
        usd_path: str,
        flip_joints: List[str] = [],
        flip_joint_limits: List[str] = [],
        usd_robot_root: str = "robot",
        tensor_args: TensorDeviceType = TensorDeviceType(),
        extra_links: Optional[Dict[str, LinkParams]] = None,
    ) -> None:
        """Initialize instance with USD file path.

        Args:
            usd_path: path to usd reference. This will opened as a Usd Stage.
            flip_joints: list of joint names to flip axis. This is required as current
                implementation does not read transformations from joint to link correctly.
            flip_joint_limits: list of joint names to flip joint limits.
            usd_robot_root: Root prim of the robot in the Usd Stage.
            tensor_args: Device and floating point precision for tensors.
            extra_links: Additional links to add to the robot kinematics structure.
        """

        # create a usd stage
        self._flip_joints = flip_joints
        self._flip_joint_limits = flip_joint_limits
        self._stage = Usd.Stage.Open(usd_path)
        self._usd_robot_root = usd_robot_root
        self._parent_joint_map = {}
        self.tensor_args = tensor_args
        super().__init__(extra_links)

    @property
    def robot_prim_root(self):
        """Root prim of the robot in the Usd Stage."""
        return self._usd_robot_root

    def build_link_parent(self):
        """Build a dictionary containing parent link for each link in the robot."""
        self._parent_map = {}
        all_joints = [
            x
            for x in self._stage.Traverse()
            if (
                x.IsA(UsdPhysics.Joint)
                and str(x.GetPath()).startswith(self._usd_robot_root)
            )
        ]
        for l in all_joints:
            parent, child = get_links_for_joint(l)
            if child is not None and parent is not None:
                self._parent_map[child.GetName()] = {"parent": parent.GetName()}
                self._parent_joint_map[child.GetName()] = l  # store joint prim

    def get_link_parameters(self, link_name: str, base: bool = False) -> LinkParams:
        """Get Link parameters from usd stage.

        Includes logic to detect flipped axes and apply joint_offset=[-1.0, 0.0],
        similar to how UrdfKinematicsParser handles negative axes.
        """
        link_params = self._get_from_extra_links(link_name)
        if link_params is not None:
            return link_params

        joint_limits = None
        joint_offset = [1.0, 0.0]  # Default: Multiplier 1.0, Offset 0.0

        if base:
            parent_link_name = None
            joint_transform = np.eye(4)
            joint_name = "base_joint"
            joint_type = JointType.FIXED

        else:
            parent_link_name = self._parent_map[link_name]["parent"]
            joint_prim = self._parent_joint_map[link_name]  # joint prim connects link
            joint_transform = self._get_joint_transform(joint_prim)
            joint_axis = None
            joint_name = joint_prim.GetName()

            # --- [Axis Flip Detection Logic] ---
            # 1. Get Rot1 (Child -> Joint transform)
            j_api = UsdPhysics.Joint(joint_prim)
            quat1_gf = j_api.GetLocalRot1Attr().Get()

            # 2. Get Defined Axis
            raw_axis = "X"
            if joint_prim.IsA(UsdPhysics.RevoluteJoint):
                raw_axis = UsdPhysics.RevoluteJoint(joint_prim).GetAxisAttr().Get()
            elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
                raw_axis = UsdPhysics.PrismaticJoint(joint_prim).GetAxisAttr().Get()

            # 3. Create Axis Vector (Must be Vec3f)
            axis_vec = Gf.Vec3f(0, 0, 0)
            if raw_axis == "X":
                axis_vec[0] = 1.0
            elif raw_axis == "Y":
                axis_vec[1] = 1.0
            elif raw_axis == "Z":
                axis_vec[2] = 1.0

            # 4. Check alignment in Child Frame
            transformed_axis = quat1_gf.GetInverse().Transform(axis_vec)

            dot_product = 0.0
            if raw_axis == "X":
                dot_product = transformed_axis[0]
            elif raw_axis == "Y":
                dot_product = transformed_axis[1]
            elif raw_axis == "Z":
                dot_product = transformed_axis[2]

            # Auto-detect flip (if dot product is negative)
            is_flipped_auto = dot_product < -0.9

            # Manual override check
            is_flipped_manual = False
            if isinstance(self._flip_joints, list):
                is_flipped_manual = joint_name in self._flip_joints
            elif isinstance(self._flip_joints, dict):
                is_flipped_manual = joint_name in self._flip_joints.keys()

            should_flip = is_flipped_auto or is_flipped_manual

            # --- [Apply Flip via Offset] ---
            if should_flip:
                # Instead of rotating the frame, we invert the multiplier.
                # This matches URDF parser logic for negative axes.
                joint_offset[0] = -1.0

            # --- [Joint Type Parsing] ---
            if joint_prim.IsA(UsdPhysics.FixedJoint):
                joint_type = JointType.FIXED

            elif joint_prim.IsA(UsdPhysics.RevoluteJoint):
                j_prim = UsdPhysics.RevoluteJoint(joint_prim)
                joint_axis = raw_axis

                lower = j_prim.GetLowerLimitAttr().Get()
                upper = j_prim.GetUpperLimitAttr().Get()

                # If flipped, we should also swap limits for correctness
                if should_flip:
                    temp = lower
                    lower = -upper
                    upper = -temp

                joint_limits = np.radians(np.ravel([lower, upper]))

                if joint_axis == "X":
                    joint_type = JointType.X_ROT
                elif joint_axis == "Y":
                    joint_type = JointType.Y_ROT
                elif joint_axis == "Z":
                    joint_type = JointType.Z_ROT
                else:
                    log_error("Joint axis not supported" + str(joint_axis))

            elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
                j_prim = UsdPhysics.PrismaticJoint(joint_prim)
                joint_axis = raw_axis

                lower = j_prim.GetLowerLimitAttr().Get()
                upper = j_prim.GetUpperLimitAttr().Get()

                # Manual limit flip override + Auto flip
                manual_limit_flip = joint_name in self._flip_joint_limits
                if should_flip or manual_limit_flip:
                    temp = lower
                    lower = -upper
                    upper = -temp

                # Prismatic does NOT use radians
                joint_limits = np.ravel([lower, upper])

                if joint_axis == "X":
                    joint_type = JointType.X_PRISM
                elif joint_axis == "Y":
                    joint_type = JointType.Y_PRISM
                elif joint_axis == "Z":
                    joint_type = JointType.Z_PRISM
                else:
                    log_error("Joint axis not supported" + str(joint_axis))
            else:
                # Default fallback or error
                log_error(f"Joint type not supported: {joint_prim.GetTypeName()}")
                joint_type = JointType.FIXED

        link_params = LinkParams(
            link_name=link_name,
            joint_name=joint_name,
            joint_type=joint_type,
            fixed_transform=joint_transform,
            parent_link_name=parent_link_name,
            joint_limits=joint_limits,
            joint_offset=joint_offset,  # Passed to CuRobo LinkParams
        )
        return link_params

    def _get_joint_transform(self, prim: Usd.Prim) -> Pose:
        """Get pose of link from joint prim.

        Args:
            prim: joint prim in the usd stage.

        Returns:
            Pose: pose of the link from joint origin.
        """
        j_prim = UsdPhysics.Joint(prim)
        position = np.ravel(j_prim.GetLocalPos0Attr().Get())
        quatf = j_prim.GetLocalRot0Attr().Get()
        quat = np.zeros(4)
        quat[0] = quatf.GetReal()
        quat[1:] = quatf.GetImaginary()

        # create a homogenous transformation matrix:
        transform_0 = Pose(
            self.tensor_args.to_device(position), self.tensor_args.to_device(quat)
        )

        position = np.ravel(j_prim.GetLocalPos1Attr().Get())
        quatf = j_prim.GetLocalRot1Attr().Get()
        quat = np.zeros(4)
        quat[0] = quatf.GetReal()
        quat[1:] = quatf.GetImaginary()

        # create a homogenous transformation matrix:
        transform_1 = Pose(
            self.tensor_args.to_device(position), self.tensor_args.to_device(quat)
        )
        transform = (
            transform_0.multiply(transform_1.inverse())
            .get_matrix()
            .cpu()
            .view(4, 4)
            .numpy()
        )

        # get attached link transform:

        return transform


def get_links_for_joint(
    prim: Usd.Prim,
) -> Tuple[Optional[Usd.Prim], Optional[Usd.Prim]]:
    """Get all link prims from the given joint prim.


    This assumes that the `body0_rel_targets` and `body1_rel_targets` are configured such
    that the parent link is specified in `body0_rel_targets` and the child links is specified
    in `body1_rel_targets`.

    Args:
        prim: joint prim in the usd stage.

    Returns:
        Tuple[Optional[Usd.Prim], Optional[Usd.Prim]]: parent link prim and child link prim.
    """
    stage = prim.GetStage()
    joint_api = UsdPhysics.Joint(prim)

    rel0_targets = joint_api.GetBody0Rel().GetTargets()
    if len(rel0_targets) > 1:
        raise NotImplementedError(
            "`get_links_for_joint` does not currently handle more than one relative"
            f" body target in the joint. joint_prim: {prim}, body0_rel_targets:"
            f" {rel0_targets}"
        )
    link0_prim = None
    if len(rel0_targets) != 0:
        link0_prim = stage.GetPrimAtPath(rel0_targets[0])

    rel1_targets = joint_api.GetBody1Rel().GetTargets()
    if len(rel1_targets) > 1:
        raise NotImplementedError(
            "`get_links_for_joint` does not currently handle more than one relative"
            f" body target in the joint. joint_prim: {prim}, body1_rel_targets:"
            f" {rel0_targets}"
        )
    link1_prim = None
    if len(rel1_targets) != 0:
        link1_prim = stage.GetPrimAtPath(rel1_targets[0])

    return (link0_prim, link1_prim)
