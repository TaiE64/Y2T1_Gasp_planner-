from abc import ABC, abstractmethod
import pybullet as p
import pandas as pd
import numpy as np
import time
import random
import os


class Gripper(ABC):
    # Create dict to save data
    data = {'Ori_x': [],  # gripper orientation(base)
            'Ori_y': [],
            'Ori_z': [],
            'Ori_w': [],
            'Pos_x': [],  # gripper position(base)
            'Pos_y': [],
            'Pos_z': [],
            'Obj_Pos_x': [],  # object position
            'Obj_Pos_y': [],
            'Obj_Pos_z': [],
            'Label': []  # Good or Bad grasp (1 / 0)
            }
    file_path = "data.csv"
    # Initial position and orientation to load hand
    initial_position = [0, 0, 3]
    initial_orientation = p.getQuaternionFromEuler([0, 0, 0])

    def __init__(self, where, objectname, batch_size, gripperType):
        self.where = where
        self.objectname = objectname
        self.batch_size = batch_size
        self.gripperType = gripperType

    def loadObj(self, objectname):
        # Loading different object to be graspped
        if objectname == 'cube':
            objID = p.loadURDF("urdf\Object\cube\cube_small.urdf", [
                               0.0, -0.4, 0.62], globalScaling=2)
            return objID
        elif objectname == 'dog':
            objID = p.loadURDF(
                "urdf/Object/dog/a1_description/urdf/a1_further_scaled_with_friction.urdf", [0.0, -0.4, 1], globalScaling=2)
            return objID
        elif objectname == 'duck':
            objID = p.loadURDF("urdf\Object\duck\duck_vhacd.urdf", [
                               0.0, -0.4, 1], globalScaling=2)
            return objID
        elif objectname == 'mug':
            objID = p.loadURDF("urdf\Object\mug\mug.urdf", [
                               0.0, -0.4, 1], globalScaling=2)
            return objID
        else:
            print("We dont have this object...")

    def loadTable(self):
        # Loading table
        table = p.loadURDF("urdf/Table/table.urdf",
                           [0.0, -0.4, 0], [0, 0, 0, 1], 0)

    def random_position(self, object_id, where):
        # Generate random position around object to grasp
        aabb = p.getAABB(object_id)
        min_bound, max_bound = aabb
        print(f'Object\'s Min Bound: {min_bound}')
        print(f'Object\'s Max Bound: {max_bound}')

        if where == 0:
            # noise for above(0)
            noise_1 = random.gauss(mu=0, sigma=0.3)*0.1  # x noise
            noise_2 = random.gauss(mu=0, sigma=0.3)*0.1  # y noise
            noise_3 = random.gauss(mu=0.05, sigma=0.05)  # z noise
            # Generate random position above the object (0)
            x_safe = (max_bound[0]+min_bound[0])/2+noise_1
            y_safe = (max_bound[1]+min_bound[1])/2+noise_2
            z_safe = max_bound[2]+noise_3

        elif where == 1:
            # noise for side(1)
            noise_1 = random.gauss(mu=0.05, sigma=0.05)  # x noise
            noise_2 = random.gauss(mu=0, sigma=0.5)*0.1  # y noiseA
            noise_3 = random.gauss(mu=-0.05, sigma=0.5)*0.1  # z noise
            # Generate random position from the side(1)
            x_safe = min_bound[0]-noise_1
            y_safe = (max_bound[1]+min_bound[1])/2+noise_2
            z_safe = (max_bound[2]+min_bound[2])/2+noise_3

        elif where == 2:
            # noise for side(2)
            noise_1 = random.gauss(mu=0, sigma=0.5)*0.1  # x noise
            noise_2 = random.gauss(mu=0.05, sigma=0.05)  # y noiseA
            noise_3 = random.gauss(mu=-0.05, sigma=0.5)*0.1
            # Generate random position from the side(2)
            x_safe = (max_bound[0]+min_bound[0])/2+noise_1
            y_safe = min_bound[1]-noise_2
            z_safe = (max_bound[2]+min_bound[2])/2+noise_3

        elif where == 3:
            # noise for side(3)
            noise_1 = random.gauss(mu=0, sigma=0.5)*0.1  # x noise
            noise_2 = random.gauss(mu=0.05, sigma=0.05)  # y noiseA
            noise_3 = random.gauss(mu=-0.05, sigma=0.5)*0.1  # z noise
            # Generate random position from the side(3)
            x_safe = (max_bound[0]+min_bound[0])/2+noise_1
            y_safe = max_bound[1]+noise_2
            z_safe = (max_bound[2]+min_bound[2])/2+noise_3

        elif where == 4:
            # noise for side(4)
            noise_1 = random.gauss(mu=0.05, sigma=0.05)  # x noise
            noise_2 = random.gauss(mu=0, sigma=0.5)*0.1  # y noiseA
            noise_3 = random.gauss(mu=-0.05, sigma=0.5)*0.1  # z noise
            # Generate random position from the side(4)
            x_safe = max_bound[0]+noise_1
            y_safe = (max_bound[1]+min_bound[1])/2+noise_2
            z_safe = (max_bound[2]+min_bound[2])/2+noise_3

        else:
            print(
                "Please input number 0 ~ 4 to select the direction to generate random grasp")

        return [x_safe, y_safe, z_safe]

    def update_gripper_direction(self, safe_position, gripperCID, where, gripperType):
        #  Slowly move over (prevent collision with object)
        # Current gripper pos
        current_position = [0, 0, 3]
        if gripperType == 'PR2':
            if where == 0:
                hand_ori = p.getQuaternionFromEuler(
                    [0, np.pi/2, 0])  # #above the object  (0)
            elif where == 1:
                hand_ori = p.getQuaternionFromEuler(
                    [0, 0, 0])  # at side of the object (1)
            elif where == 2:
                hand_ori = p.getQuaternionFromEuler(
                    [np.pi, 0, np.pi/2])  # at side of the object  (2)
            elif where == 3:
                hand_ori = p.getQuaternionFromEuler(
                    [np.pi, 0, -np.pi/2])  # at side of the object (3)
            elif where == 4:
                hand_ori = p.getQuaternionFromEuler(
                    [0, 0, np.pi])  # at side of the object  (4)
            steps = 80
            for step in range(steps):
                t = step / steps
                interpolated_position = [
                    current_position[i] + (safe_position[i] - current_position[i]) * t for i in range(3)]
                # Update Constraint: Position and Direction
                p.changeConstraint(gripperCID, interpolated_position,
                                   jointChildFrameOrientation=hand_ori)
                p.stepSimulation()
                time.sleep(0.01)
            print("Gripper position updated gradually.")
            time.sleep(0.4)
        elif gripperType == 'F3':
            if where == 0:
                hand_ori = p.getQuaternionFromEuler(
                    [0, -np.pi, np.pi/2])  # #above the object  (0)
            elif where == 1:
                hand_ori = p.getQuaternionFromEuler(
                    [np.pi/2, 0, np.pi/2])  # at side of the object (1)
            elif where == 2:
                hand_ori = p.getQuaternionFromEuler(
                    [np.pi/2, 0, np.pi])  # at side of the object  (2)
            elif where == 3:
                hand_ori = p.getQuaternionFromEuler(
                    [-np.pi/2, 0, np.pi])  # at side of the object (3)
            elif where == 4:
                hand_ori = p.getQuaternionFromEuler(
                    [-np.pi/2, 0, np.pi/2])  # at side of the object  (4)
            steps = 80
            for step in range(steps):
                t = step / steps
                interpolated_position = [
                    current_position[i] + (safe_position[i] - current_position[i]) * t for i in range(3)]
                # Update Constraint: Position and Direction
                p.changeConstraint(gripperCID, interpolated_position,
                                   jointChildFrameOrientation=hand_ori)
                p.stepSimulation()
                time.sleep(0.01)
            print("Gripper position updated gradually.")
            time.sleep(0.4)

    def set_camera(self):
        # Set view of simulation window
        camera_distance = 2
        camera_yaw = 45
        camera_pitch = -30
        target_position = [0.0, -0.4, 0.5]

        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=target_position
        )

    def getdata(self, object_id, gripperID):
        # collect data : position and orientation of gripper after grasp the object
        obj_pos, _ = p.getBasePositionAndOrientation(object_id)
        print(obj_pos)
        print()
        print('_______data_______')
        position, orientation = p.getBasePositionAndOrientation(gripperID)
        print(f'Position of gripper:{position}')
        print(f'Orientation of gripper:{orientation}')
        print(f'Position of object:{obj_pos}')
        print('_______data_______')
        print()
        return position, orientation, obj_pos

    def SaveData(self, position, orientation, label, obj_pos):
        # Saving data : Gripper Position,Gripper Orientation,Object Position,Label(Quality of grasp)
        Gripper.data['Ori_x'].append(round(orientation[0], 6))
        Gripper.data['Ori_y'].append(round(orientation[1], 6))
        Gripper.data['Ori_z'].append(round(orientation[2], 6))
        Gripper.data['Ori_w'].append(round(orientation[3], 6))
        Gripper.data['Pos_x'].append(round(position[0], 6))
        Gripper.data['Pos_y'].append(round(position[1], 6))
        Gripper.data['Pos_z'].append(round(position[2], 6))
        Gripper.data['Obj_Pos_x'].append(round(obj_pos[0], 6))
        Gripper.data['Obj_Pos_y'].append(round(obj_pos[1], 6))
        Gripper.data['Obj_Pos_z'].append(round(obj_pos[2], 6))
        Gripper.data['Label'].append(label)

        df = pd.DataFrame(Gripper.data)
        if os.path.exists(Gripper.file_path):
            df.to_csv(Gripper.file_path, mode="a", header=False, index=False)
            print("Data appended to existing file")
        else:
            # Create and write header
            df.to_csv(Gripper.file_path, index=False)
            print("File created and data saved")
        Gripper.data = {key: [] for key in Gripper.data.keys()}

    def is_successful(self, object_id, gripperID):
        # Label of the grasping : 0 or 1
        # if grasp can hold for 3 seconds
        Label = None
        print('Evaluating grasp quality....')
        time.sleep(3)
        contact_points = p.getContactPoints(bodyA=gripperID, bodyB=object_id)
        if len(contact_points) == 0:
            Label = 0
            print('Bad grasp :<')
        elif len(contact_points) >= 2:
            Label = 1
            print('Success! :>')
        else:
            print("Something went wrong...")
        print()
        return Label

    def lifting(self, safe_position, gripperCID):  # CID = Constraint ID
        # Lift object up slowly to qualify grasp quality
        print("Lifting....")
        for step in range(100):
            target_z = safe_position[2] + step * 0.005
            p.changeConstraint(
                gripperCID, [safe_position[0], safe_position[1], target_z])
            p.stepSimulation()
            time.sleep(0.01)
        print()

    def random_grasp_object(self, object_id, gripperID, gripperCID, where, gripperType):
        # procedure of grasping and collect data
        time.sleep(2)  # Allow time for the object to stabilize
        self.openGripper(gripperID)
        # Generate random position around the object
        random_Pos = self.random_position(object_id, where)

        # update gripper Postion to random_Pos and predefined Orientation
        self.update_gripper_direction(
            random_Pos, gripperCID, where, gripperType)

        # Reshape pos, get ready for grasping
        self.reshape(gripperID)

        # Grasp object at random position
        self.grasp(gripperID)

        # Get base link's(wrist of the gripper,link 0) Position and Orientation
        position, orientation, obj_pos = self.getdata(object_id, gripperID)

        # Lifting object
        self.lifting(random_Pos, gripperCID)

        # Qualify grasping
        label = self.is_successful(object_id, gripperID)

        # Saving data
        self.SaveData(position, orientation, label, obj_pos)

        # print(data) #check data

# Define Abstractmethods that need to be implemented by each specific type of gripper
# Grasping method
    @abstractmethod
    def openGripper(self, gripperID):
        pass

    @abstractmethod
    def reshape(self, gripperID):
        pass

    @abstractmethod
    def grasp(self, gripperID):
        pass

# Data collectionm method
    @abstractmethod
    def Debug_GUI(self):
        pass

    @abstractmethod
    def DataCollection(self):
        pass


class PR2(Gripper):
    def __init__(self, where, objectname, batch_size, gripperType):
        super().__init__(where, objectname, batch_size, gripperType)

    def openGripper(self, gripperID):
        jointPositions = [3, 0.000000, 3, 0.000000]
        for jointIndex in range(p.getNumJoints(gripperID)):
            p.resetJointState(gripperID, jointIndex,
                              jointPositions[jointIndex])

    def reshape(self, gripperID):
        p.setJointMotorControl2(gripperID, 0, p.POSITION_CONTROL,
                                targetPosition=0.05, maxVelocity=1, force=1)
        p.setJointMotorControl2(gripperID, 2, p.POSITION_CONTROL,
                                targetPosition=0.05, maxVelocity=1, force=1)

    def grasp(self, gripperID):
        p.setJointMotorControl2(gripperID, 0, p.POSITION_CONTROL,
                                targetPosition=0, maxVelocity=5, force=2000)
        p.setJointMotorControl2(gripperID, 2, p.POSITION_CONTROL,
                                targetPosition=0, maxVelocity=5, force=2000)
        time.sleep(2)

    def Debug_GUI(self):
        # --------------------Initialization Start--------------------
        cid = p.connect(p.SHARED_MEMORY)
        if (cid < 0):
            p.connect(p.GUI)
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.setRealTimeSimulation(1)
        p.setGravity(0, 0, -10)
        gripperID = [p.loadURDF("urdf\Gripper\pr2\pr2_gripper.urdf",
                                Gripper.initial_position, Gripper.initial_orientation)][0]
        self.loadTable()
        objID = self.loadObj(self.objectname)
        pr2_cid = p.createConstraint(gripperID, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                     [0, 0, 0])  # Create constraint of gripper Gripper CID
        # --------------------Initialization End--------------------

        # --------------------Start Simulation--------------------
        os.system('cls')  # Clear terminal
        print("------------------------------Start------------------------------")
        self.random_grasp_object(
            objID, gripperID, pr2_cid, self.where, self.gripperType)
        p.stepSimulation()
        print("------------------------------End------------------------------")
        time.sleep(2)
        input("Enter to quit")
        p.disconnect()

    def DataCollection(self):
        # --------------------Initialization Start--------------------
        for i in range(self.batch_size):
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                p.connect(p.GUI)
            p.resetSimulation()
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            gripperID = [p.loadURDF("urdf\Gripper\pr2\pr2_gripper.urdf",
                                    Gripper.initial_position, Gripper.initial_orientation)][0]
            self.loadTable()
            objID = self.loadObj(self.objectname)
            pr2_cid = p.createConstraint(gripperID, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                         [0, 0, 0])  # Create constraint of gripper Gripper CID
            self.set_camera()
            # --------------------Initialization End--------------------

            # --------------------Start Simulation--------------------
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            p.setRealTimeSimulation(1)
            p.setGravity(0, 0, -10)

            # --------------------Start Data Colletion--------------------
            os.system('cls')  # Clear terminal
            print(f"--- Collecting Cample {i + 1} / {self.batch_size} ---")
            self.random_grasp_object(
                objID, gripperID, pr2_cid, self.where, self.gripperType)
            p.stepSimulation()
            print("--- Data Collection Completed ---")
            time.sleep(2)
            p.disconnect()


class F3(Gripper):
    def __init__(self, where, objectname, batch_size, gripperType):
        super().__init__(where, objectname, batch_size, gripperType)

    def openGripper(self, gripperID):
        numJoints = p.getNumJoints(gripperID)
        joints = []
        for i in range(0, numJoints):
            joints.append(p.getJointState(gripperID, i)[0])
        for k in range(0, numJoints):
            # lower finger joints
            if k == 2 or k == 5 or k == 8:
                p.setJointMotorControl2(gripperID, k, p.POSITION_CONTROL,
                                        targetPosition=0,
                                        maxVelocity=5, force=2000)
            # Upper finger joints
            elif k == 6 or k == 3 or k == 9:
                p.setJointMotorControl2(gripperID, k, p.POSITION_CONTROL,
                                        targetPosition=0,
                                        maxVelocity=5, force=2000)
            # Base finger joints
            elif k == 1 or k == 4 or k == 7:
                p.setJointMotorControl2(gripperID, k, p.POSITION_CONTROL,
                                        targetPosition=-3,
                                        maxVelocity=5, force=2000)

    def reshape(self, gripperID):
        numJoints = p.getNumJoints(gripperID)
        joints = []
        for i in range(0, numJoints):
            joints.append(p.getJointState(gripperID, i)[0])
        for k in range(0, numJoints):
            # lower finger joints
            if k == 2 or k == 5 or k == 8:
                p.setJointMotorControl2(gripperID, k, p.POSITION_CONTROL,
                                        targetPosition=0.5,
                                        maxVelocity=5, force=2000)
            # Upper finger joints
            elif k == 6 or k == 3 or k == 9:
                p.setJointMotorControl2(gripperID, k, p.POSITION_CONTROL,
                                        targetPosition=0,
                                        maxVelocity=5, force=2000)
            # Base finger joints
            elif k == 1 or k == 4 or k == 7:
                p.setJointMotorControl2(gripperID, k, p.POSITION_CONTROL,
                                        targetPosition=-3,
                                        maxVelocity=5, force=2000)

    def grasp(self, gripperID):
        numJoints = p.getNumJoints(gripperID)
        for k in range(0, numJoints):
            # lower finger joints
            if k == 2 or k == 5 or k == 8:
                p.setJointMotorControl2(gripperID, k, p.POSITION_CONTROL,
                                        targetPosition=0.5,
                                        maxVelocity=5, force=200)
            # Upper finger joints
            elif k == 6 or k == 3 or k == 9:
                p.setJointMotorControl2(gripperID, k, p.POSITION_CONTROL,
                                        targetPosition=0,
                                        maxVelocity=5, force=200)
            # Base finger joints
            elif k == 1 or k == 4 or k == 7:
                p.setJointMotorControl2(gripperID, k, p.POSITION_CONTROL,
                                        targetPosition=-0.3,
                                        maxVelocity=5, force=200)
        time.sleep(2)

    def Debug_GUI(self):
        # --------------------Initialization Start--------------------
        cid = p.connect(p.SHARED_MEMORY)
        if (cid < 0):
            p.connect(p.GUI)
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.setRealTimeSimulation(1)
        p.setGravity(0, 0, -10)
        gripperID = [p.loadURDF("urdf\Gripper/3fingers\sdh_modified.urdf",
                                [0, 0, 1], Gripper.initial_orientation, globalScaling=1)][0]
        num_joints = p.getNumJoints(gripperID)
        all_links = [-1] + [i for i in range(num_joints)]
        for link_id in all_links:
            p.changeDynamics(
                bodyUniqueId=gripperID,
                linkIndex=link_id,
                lateralFriction=10,
                spinningFriction=10
            )
        self.loadTable()
        objID = self.loadObj(self.objectname)
        f3_cid = p.createConstraint(gripperID, -1, -1, -1, p.JOINT_FIXED,
                                    [0, 0, 0], [0, 0, 0], Gripper.initial_orientation)

        # --------------------Initialization End--------------------

        # --------------------Start Simulation--------------------
        os.system('cls')  # Clear terminal
        print("------------------------------Start------------------------------")
        self.random_grasp_object(
            objID, gripperID, f3_cid, self.where, self.gripperType)
        p.stepSimulation()
        print("------------------------------End------------------------------")
        time.sleep(2)
        input("Enter to quit")
        p.disconnect()

    def DataCollection(self):
        # --------------------Initialization Start--------------------
        for i in range(self.batch_size):
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                p.connect(p.GUI)  # DIRECT/GUI
            p.resetSimulation()
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            gripperID = [p.loadURDF("urdf/Gripper/3fingers/sdh.urdf",
                                    [0, 0, 1], Gripper.initial_orientation, globalScaling=1)][0]
            num_joints = p.getNumJoints(gripperID)
            all_links = [-1] + [i for i in range(num_joints)]
            for link_id in all_links:
                p.changeDynamics(
                    bodyUniqueId=gripperID,
                    linkIndex=link_id,
                    lateralFriction=10,
                    spinningFriction=10
                )
            self.loadTable()
            objID = self.loadObj(self.objectname)
            pr2_cid = p.createConstraint(gripperID, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                         [0, 0, 0])
            self.set_camera()
            # --------------------Initialization End--------------------

            # --------------------Start Simulation--------------------
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            p.setRealTimeSimulation(1)
            p.setGravity(0, 0, -10)

            # --------------------Start Data Colletion--------------------
            os.system('cls')  # Clear terminal
            print(f"--- Collecting Cample {i + 1} / {self.batch_size} ---")
            self.random_grasp_object(
                objID, gripperID, pr2_cid, self.where, self.gripperType)
            p.stepSimulation()
            print("--- Data Collection Completed ---")
            time.sleep(2)
            p.disconnect()

# Select: Gripper Type, Direction to Grasp,Object,DataCollection Batch Size,operation mode


class RUN():
    def __init__(self, gripperType, where, objectname, batch_size=100, GUI=0):
        if gripperType == 'PR2':
            a = PR2(where, objectname, batch_size, gripperType)
        elif gripperType == 'F3':
            a = F3(where, objectname, batch_size, gripperType)

        if GUI == 0:
            a.Debug_GUI()
        elif GUI == 1:
            a.DataCollection()
        else:
            print("Please enter number 0 or 1 to select operation mode")

# Select: Gripper Type, Direction to Grasp,Object,DataCollection Batch Size,operation mode
# GripperType：PR2,F3
# Direction to grasp: 0 ~ 4
# Object : cube, dog, duck, mug
# operation mode:
# 0 → Run to see the grasping process , for debugging
# 1 → Run to repetitively collect data


if __name__ == '__main__':
    a = RUN('F3', 0, 'cube', 500, 0)
