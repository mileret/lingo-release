import bpy
import numpy as np
from mathutils import Vector, Quaternion

SMPLX_JOINT_NAMES = [
    'pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle','right_ankle','spine3', 'left_foot','right_foot','neck','left_collar','right_collar','head','left_shoulder','right_shoulder','left_elbow', 'right_elbow','left_wrist','right_wrist',
    'jaw','left_eye_smplhf','right_eye_smplhf','left_index1','left_index2','left_index3','left_middle1','left_middle2','left_middle3','left_pinky1','left_pinky2','left_pinky3','left_ring1','left_ring2','left_ring3','left_thumb1','left_thumb2','left_thumb3','right_index1','right_index2','right_index3','right_middle1','right_middle2','right_middle3','right_pinky1','right_pinky2','right_pinky3','right_ring1','right_ring2','right_ring3','right_thumb1','right_thumb2','right_thumb3'
]
NUM_SMPLX_JOINTS = len(SMPLX_JOINT_NAMES)
NUM_SMPLX_BODYJOINTS = 21
NUM_SMPLX_HANDJOINTS = 15


def animation_data_clear(obj):

    obj.animation_data_clear()
    obj.data.animation_data_clear()


def load_smplx_animation(file, obj):
    animation_data_clear(obj)

    armature = obj.parent
    bpy.context.view_layer.objects.active = obj  # mesh needs to be active object for recalculating joint locations

    with np.load("./smplx_handposes.npz", allow_pickle=True) as data:
        hand_poses = data["hand_poses"].item()
        (left_hand_pose, right_hand_pose) = hand_poses["relaxed"]
        hand_pose_relaxed = np.concatenate((left_hand_pose, right_hand_pose)).reshape(NUM_SMPLX_HANDJOINTS * 2, 3)

    translation = file["transl"].reshape(-1, 3)
    global_orient = file["global_orient"].reshape(-1, 3)
    body_pose = np.array(file["body_pose"]).reshape(-1, NUM_SMPLX_BODYJOINTS, 3)
    left_hand_pose = np.zeros((len(body_pose), NUM_SMPLX_HANDJOINTS, 3))
    right_hand_pose = np.zeros((len(body_pose), NUM_SMPLX_HANDJOINTS, 3))
    betas = [0.0] * 10
    num_keyframes = len(body_pose)

    bpy.ops.object.mode_set(mode='OBJECT')
    for index, beta in enumerate(betas):
        key_block_name = f"Shape{index:03}"

        if key_block_name in obj.data.shape_keys.key_blocks:
            obj.data.shape_keys.key_blocks[key_block_name].value = beta
        else:
            print(f"ERROR: No key block for: {key_block_name}")

    bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')

    global_orients = np.array([get_quat_from_rodrigues(rodrigues) for rodrigues in global_orient])
    body_poses = {bone_name: np.array([get_quat_from_rodrigues(rodrigues) for rodrigues in body_pose[:, index, :]])
                  for index, bone_name in enumerate(SMPLX_JOINT_NAMES[1: NUM_SMPLX_BODYJOINTS + 1])}

    start_name_index = 1 + NUM_SMPLX_BODYJOINTS + 3
    left_hand_poses = {bone_name: np.array([get_quat_from_rodrigues(rodrigues, hand_pose_relaxed[i]) for rodrigues in left_hand_pose[:, i, :]])
                       for i, bone_name in enumerate(SMPLX_JOINT_NAMES[start_name_index: start_name_index + NUM_SMPLX_HANDJOINTS])}

    start_name_index = 1 + NUM_SMPLX_BODYJOINTS + 3 + NUM_SMPLX_HANDJOINTS
    right_hand_poses = {bone_name: np.array([get_quat_from_rodrigues(rodrigues, hand_pose_relaxed[NUM_SMPLX_HANDJOINTS + i]) for rodrigues in right_hand_pose[:, i, :]])
                        for i, bone_name in enumerate(SMPLX_JOINT_NAMES[start_name_index: start_name_index + NUM_SMPLX_HANDJOINTS])}

    body_poses = {**body_poses, **left_hand_poses, **right_hand_poses}
    body_poses['pelvis'] = global_orients

    animation_data = armature.animation_data_create()
    action = animation_data.action = bpy.data.actions.new(f'{armature.name}Action')
    for i in range(3):
        fcurve = action.fcurves.new('pose.bones["root"].location', index=i)
        fcurve.keyframe_points.add(count=num_keyframes)
        fcurve.keyframe_points.foreach_set("co", [x for co in zip(range(num_keyframes), translation[:, i]) for x in co])
        fcurve.update()
    for bone_name, quaternions in body_poses.items():
        for i in range(4):
            fcurve = action.fcurves.new(f'pose.bones["{bone_name}"].rotation_quaternion', index=i)
            fcurve.keyframe_points.add(count=num_keyframes)
            fcurve.keyframe_points.foreach_set("co",
                                               [x for co in zip(range(num_keyframes), quaternions[:, i]) for x in co])
            fcurve.update()

    bpy.context.scene.frame_set(0)

    # Activate corrective poseshapes
    bpy.ops.object.smplx_set_poseshapes('EXEC_DEFAULT')

    return {'FINISHED'}


def get_quat_from_rodrigues(rodrigues, rodrigues_reference=None):
    rod = Vector((rodrigues[0], rodrigues[1], rodrigues[2]))
    angle_rad = rod.length
    axis = rod.normalized()

    quat = Quaternion(axis, angle_rad)

    if rodrigues_reference is None:
       return quat
    else:
        rod_reference = Vector((rodrigues_reference[0], rodrigues_reference[1], rodrigues_reference[2]))
        rod_result = rod + rod_reference
        angle_rad_result = rod_result.length
        axis_result = rod_result.normalized()
        quat_result = Quaternion(axis_result, angle_rad_result)
        return quat_result


def set_pose_from_rodrigues(armature, bone_name, rodrigues, rodrigues_reference=None):
    rod = Vector((rodrigues[0], rodrigues[1], rodrigues[2]))
    angle_rad = rod.length
    axis = rod.normalized()

    if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'

    quat = Quaternion(axis, angle_rad)

    if rodrigues_reference is None:
        armature.pose.bones[bone_name].rotation_quaternion = quat
    else:
        # SMPL-X is adding the reference rodrigues rotation to the relaxed hand rodrigues rotation, so we have to do the same here.
        # This means that pose values for relaxed hand model cannot be interpreted as rotations in the local joint coordinate system of the relaxed hand.
        # https://github.com/vchoutas/smplx/blob/f4206853a4746139f61bdcf58571f2cea0cbebad/smplx/body_models.py#L1190
        #   full_pose += self.pose_mean
        rod_reference = Vector((rodrigues_reference[0], rodrigues_reference[1], rodrigues_reference[2]))
        rod_result = rod + rod_reference
        angle_rad_result = rod_result.length
        axis_result = rod_result.normalized()
        quat_result = Quaternion(axis_result, angle_rad_result)
        armature.pose.bones[bone_name].rotation_quaternion = quat_result

        """
        rod_reference = Vector((rodrigues_reference[0], rodrigues_reference[1], rodrigues_reference[2]))
        angle_rad_reference = rod_reference.length
        axis_reference = rod_reference.normalized()
        quat_reference = Quaternion(axis_reference, angle_rad_reference)

        # Rotate first into reference pose and then add the target pose
        armature.pose.bones[bone_name].rotation_quaternion = quat_reference @ quat
        """
    return