"""
Copyright (c) 2025 The uos_sess6072_build Authors.
Authors: Blair Thornton, Alec O'Loughlin, Miquel Massot
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import numpy as np
import json
from datetime import datetime
import argparse
import time
from pathlib import Path
import subprocess
import platform

from zeroros import Publisher, Subscriber
from zeroros.messages import String, Vector3, Vector3Stamped, Pose, PoseStamped
from zeroros.datalogger import DataLogger

from drivers.aruco import ArUcoUDPDriver
from drivers.rpi import Console, Rate
from drivers import __version__
from scipy.spatial.transform import Rotation as R
from math_sess6072 import *
from model_sess6072 import *
# enter additional library imports here



# define global variables 
N = 0
E = 1
G = 2
DOTN = 3
DOTE = 4
DOTG = 5

# define global functions
def get_wifi_name():
    result = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        if "SSID" in line and "BSSID" not in line:
            return line.split(":")[1].strip()
    return 0

def Vector(dim): return np.zeros((dim, 1), dtype=float)



def rpm2N(x, fwd_lim = 2000, rev_lim = -2000): 
    tol = 10
    if x>fwd_lim: x=fwd_lim
    if x<rev_lim: x=rev_lim
    if abs(x) <= tol: return 0
    elif x>tol: return 1.541571428571430076E-7*x**2+3.293357142857142252E-4*x-1.401428571428424679E-3
    else: return -7.35749999999999954E-8*x**2+1.716749999999999581E-4*x-1.054478382732365536E-16

def N2rpm(x, fwd_lim = 1.2753, rev_lim = -0.63765): 
    tol = 10E-4
    if x>fwd_lim: x=fwd_lim
    if x<rev_lim: x=rev_lim    
    if abs(x) <= tol: return 0
    elif x>tol: return -5.727416623043567370E2*x**2+2.268233085499708977E3*x+2.958718669408357371E1
    else: return 2.397948698765131667E3*x**2+4.665568885752373717E3*x - -6.685183692128883879E-14


def feedback_control(ds, ks = None, kn = None, kg = None):
                
                if ks == None: ks = 0.1
                if kn == None: kn = 0.1
                if kg == None: kg = 0.1        
                
                dv = ks*ds[0]
                dw = kn*ds[1]+kg*ds[2]
                
                du = Vector(2)
                du[0] = dv
                du[1] = dw
                
                return du

# main class


class LaptopController:
    def __init__(self, OPERATING_MODE):
        
        ########### DEFINE ARUCO MARKER ID ###################                     
        MARKER_ID = 25 # <<< CHANGE TO YOUR ROBOT'S ARUCO ID

        ########### SET NETWORK CONDITIONS ###################             
        if OPERATING_MODE != 2: # robot
            self.robot_ip = "192.168.10.1"
            self.robot_available = False
            aruco_params = {
                "port": 50001,  # Port to listen to (DO NOT CHANGE)
                "marker_id": MARKER_ID,  # Marker ID to listen to
            }                     
            if platform.system() == "Windows": wifi_name = get_wifi_name()
            else: wifi_name = "SmartCatXX"

        elif OPERATING_MODE == 2: # webots
            self.robot_ip = "127.0.0.1"          
            aruco_params = {
                "port": 50000,  # Port to listen to (DO NOT CHANGE)
                "marker_id": 0,  # Overide for WEBOTS (DO NOT CHANGE)
            }
            wifi_name = "WEBOTS"
            self.sim_init = True # Deal with webots timestamps
                            
        Console.info("Connecting to:", self.robot_ip, "")
        if wifi_name:            
            Console.info(f"You are connected to {wifi_name}")
        else:
            Console.info("No WiFi connection detected")

        # store operating mode
        self.OPERATING_MODE = OPERATING_MODE

        ########### INITIALISE DATA LOGS ###################                     
        filename_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = Path("logs/log_" + filename_time + ".csv")
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        
        with self.filename.open('w') as f:
            f.write("EpochTime(s),TimeFromStart(s),right_prop_rate(rad/s),left_prop_rate(rad/s),LastDT(s),Yaw(rad),North(m),East(m),IMUSensedYawRate(rad/s),IMUIntegratedYaw(rad),IMUSensedTimeStamp(s),ARUCOSensedNorth(m),ARUCOSensedEast(m),ARUCOSensedYaw(rad),ArucoSensedTimeStamp(s),DepthTimeStamp(s),Depth(m)\n")      
        global file 
        file = self.filename

        ########### ENTER WAYPOINT VARIABLES ###############
        self.waypoints = None
        
        ########### INITIALISE ROBOT VARIABLES #############        
        rate = 5.0  # Hz
        self.r = Rate(rate)
        self.lastdt = 1/rate        
        self.starttime = time.time()
        self.timefromstart = None
        self.prev_sensed = None
        
        self.sensed_imu_yaw_rate_rad_s = None
        self.sensed_imu_stamp_s = None
        self.sensed_imu_prev_stamp_s = None
        self.sensed_yaw_rate = None
        self.integrated_yaw = 0

        self.sensed_pos_northings_m = None
        self.sensed_pos_eastings_m = None
        self.sensed_pos_yaw_rad = None
        self.sensed_pos_stamp_s = None
        self.sensed_bottom_depth_m = None        
        self.sensed_bottom_depth_stamp_s = None        
        
        self.initial_state = Vector(6)
        self.initial_state[N] = 0
        self.initial_state[E] = 0
        self.initial_state[G] = 0 
        self.initial_state[DOTN] = 0
        self.initial_state[DOTE] = 0
        self.initial_state[DOTG] = 0
        
        self.North = self.initial_state[N][0]
        self.East = self.initial_state[E][0]
        self.Yaw = self.initial_state[G][0]         

        self.right_rate = 0
        self.left_rate = 0
        self.lap_cpt = 0
        ############################# MOTION MODEL VARIABLES #######################
        phi=l2m([0,0])        
        x=l2m([0,0])
        y=l2m([0.09,-0.09])

        self.G=TAM(phi,x,y)
        print('G = ',self.G)
        # motion model
        # hull, water properties
        rho = 1000 # density of water in kg/m3
        draft = 0.07 #m
        beam = 0.04 #m of the immersed hull section
        length = 0.5 #m
        width = 0.4 #m # of the whole hull

        # from ESDU 71016. Fluid forces, pressures and moments on rectangular blocks. ESDU 71016 ESDU International, London
        CD = 1.5 # approximation for block from Newman (0.9 to 2.75) 
        A = 2 * beam * draft #catamaran cross section in surge
        k_drag = 0.5*rho*CD*A 

        # Added mass from Imlay 1961, Technical Report DTMB - assuming a prolate spheroid
        mass = 2.58
        e = 1 - (beam/length)**2
        alpha = (2*(1-e**2)/e**3)*(0.5*np.log((1+e)/(1-e))-e) #note np.log() = ln(), np.log10()=log()

        mass_add = 2*alpha*mass/(2-alpha)  # kg of water pushed by hull with, note this is for an infinite

        m_tot = mass + mass_add

        # Added mass from Imlay 1961, Technical Report DTMB - assuming a prolate spheroid
        I_66 = mass*((length/2)**2+(width/2)**2)/4 # rough approximation as rectangle
        e = 1 - (beam/length)**2
        alpha = (2*(1-e**2)/e**3)*(0.5*np.log((1+e)/(1-e))-e) #note np.log() = ln(), np.log10()=log()
        beta = 1/e**2 - ((1-e**2)/(2*e**3)) * np.log((1+e)/(1-e))  # kg of water pushed by hull with, note this is for an infinite

        I66_add = 2*(1/5)*mass*((draft**2-length**2)**2*(alpha-beta)/(2*(draft**2-length**2)+(draft**2+length**2)/(beta-alpha)))

        I_tot = I_66+I66_add

        # drag B_66
        B_66 = 0.12

        # read these into our vehicle class
        self.robot = Vehicle2D_e(m_tot,I_tot,k_drag,B_66)
        self.robot.info()


        self.v_robot = Vector(3) # initially stationary velocity vector in e frame
        self.p_robot = Vector(3) # pose in the e frame          
        ############################# CONTROL VARIABLES #######################
        
        
        x_path = [0,1,1]
        y_path = [0,0,1]

        # create a class instance
        trajectory = TrajectoryGenerate(x_path,y_path)

        # set velocity and acceleration constraints
        v = 0.1 #m/s
        a = 0.1/3 #takes 3s to get to 0.1m/s
        trajectory.path_to_trajectory(v, a)

        # generate turning arcs for a specified radius
        radius= 0.25
        trajectory.turning_arcs(radius)
        self.trajectory = trajectory
        self.L =0.2
        self.tau_s = 0.5
        self.kn =0
        self.kg = 0
        ############################# GUIDANCE VARIABLES ######################
        
        ############################# NAVIGATION VARIABLES ####################

        ############################# DECLARE PUBLISHERS AND SUBSCRIBERS ######         
        self.control_pub = Publisher("/control", Vector3, ip=self.robot_ip)
        self.config_pub = Publisher("/config", String, ip=self.robot_ip)
        self.imu_sub = Subscriber("/imu", Vector3, self.imu_cb, ip=self.robot_ip)
        self.sonar_sub = Subscriber("/sonar", Vector3, self.sonar_cb, ip=self.robot_ip)
        self.console_sub = Subscriber("/command", String, self.command_cb, ip=self.robot_ip)
        self.aruco_driver = ArUcoUDPDriver(aruco_params, parent=self)        
        # a callback only used by WEBOTS to fake Aruco readings 
        self.groundtruth_sub = Subscriber("/groundtruth", Pose, self.groundtruth_callback, ip=self.robot_ip) 
        self.pseudo_aruco_counter = 0


        
        ########### CONNECT TO ROBOT ###########
        if OPERATING_MODE != 2: # not a simulation
            # waits for robot to respond to configure
            count = 0
            Console.info("Connecting to robot")               
            while not self.robot_available:
                self.config_pub.publish(String("Configure"))
                time.sleep(1.0)
                count += 1
            time.sleep(5.0)
        else: # WEBOTS create fake ARUCO logs
            self.sensed_imu_stamp_s = 0 
            self.groundtruth_log = Path("logs/log_" + filename_time + "_pseudo_aruco.csv")
            with self.groundtruth_log.open('w') as f:
                f.write("epoch [s],elapsed [s],x [m],y [m],z [m],roll [deg],pitch [deg],yaw [deg],broadcast\n")

        ########### INITIALISE THRUSTERS ###########
        for i in range(10): #  rad/s
            self.control_pub.publish(Vector3())           
            self.r.sleep()

        self.initialise_pose = True # Will set to false once the pose is initialised 
        
        ######## Setup EXIT key if show_laptop not used #####
        if OPERATING_MODE == 0: # robot without show_laptop - stopped via <Ctrl+C> 
            while True:
                try:
                    self.loop()
                except KeyboardInterrupt:
                    Console.info("Ctrl+C pressed. Stopping...")
                    if self.OPERATING_MODE == 0:
                        self.imu_sub.stop()
                    break
                self.r.sleep() 
        ############################## END OF INITIALISATION ##################

    ######## DEFINE FUNCTIONS HERE ##################
    def stopcommand(self):        
        Console.info("Thrusters stopping")
        control_msg = Vector3() # initially 0
        for i in range(10):
            self.control_pub.publish(control_msg)
            self.imu_sub.stop()
            self.r.sleep()
        Console.info("Thrusters stopped")
        Console.info("Data saved in ",self.filename)
        self.r.sleep()
        
    ######## DEFINE CALLBACKS HERE ##################
    def imu_cb(self, msg: Vector3): 
        self.sensed_imu_yaw_rate_rad_s = msg.z
        self.sensed_imu_stamp_s = time.time()
        self.robot_available = True
        
    def sonar_cb(self,msg: Vector3):
        self.sensed_bottom_depth_m = msg.z/1000        
        self.sensed_bottom_depth_stamp_s = time.time()
        self.robot_available = True

    def command_cb(self,msg: String):
        Console.info(f"Response from robot: {msg.data}")

    def groundtruth_callback(self, msg):
        # generate fake aruco data at a set interval
        self.pseudo_aruco_counter += 1 

        t = time.time()
        n = msg.position.x              
        e = msg.position.y
        d = msg.position.z
        ox = msg.orientation.x
        oy = msg.orientation.y
        oz = msg.orientation.z
        ow = msg.orientation.w
        q = [ox,oy,oz,ow]                
        r = R.from_quat(q)  # note: [x, y, z, w] order
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)  # radians                
        yaw = np.mod(yaw, 360.0)

        if self.pseudo_aruco_counter==80: 
            self.pseudo_aruco_counter = 0
            self.sensed_pos_stamp_s = t
            self.sensed_pos_northings_m = n
            self.sensed_pos_eastings_m = e
            self.sensed_pos_yaw_rad = np.deg2rad(yaw)
            broadcast = True
        else:
            broadcast = False
                    
        # log groundtruth if running webots simulation
        with self.groundtruth_log.open('a') as f:
            f.write(f"{t},{t-self.starttime},{n},{e},{d},{roll},{pitch},{yaw},{broadcast}\n")

    ######## MAIN ROBOT LOOP ##################
    def loop(self):
        """This main loop is completed every 0.2 seconds.        
        Once initialised, it repeats until stopped.        
        It runs sequentially so consider how to structure your code.        
        You won't receive data from the IMU or ARUCO in every loop. 
        Don't make the loop rely on new data.
        """                
        ########## GET TIME AND DETERMINE ELAPSED TIME
        current_epoch_s = time.time()      
        self.timefromstart = current_epoch_s-self.starttime

        ######### CHECK IMU IS RUNNING OR IN WEBOTS to run loop #############
        if self.sensed_imu_stamp_s is not None or self.OPERATING_MODE == 2:  
            
            
            #----- BELOW IS A SUGGESTED STRUCTURE FOR CODE DEVELOPMENT -----#
            
            ### DETERMINE LAST THRUSTER ACTIONS ##################
            
            ### PREDICT NEW STATE ################################        
                    
            ### RECEIVE SENSOR DATA ##############################    
            
            ### RESET SENSOR DATA TO AVOID DUPLICATES ###
            self.sensed_pos_stamp_s = None
            self.sensed_pos_northings_m = None
            self.sensed_pos_eastings_m = None
            self.sensed_pos_yaw_rad = None    

            sensed_pos = self.aruco_driver.read()
            
            if sensed_pos is not None:
                self.sensed_pos_stamp_s = sensed_pos[0]
                self.sensed_pos_northings_m = sensed_pos[1]
                self.sensed_pos_eastings_m = sensed_pos[2]
                self.sensed_pos_yaw_rad = sensed_pos[6]                
                print(
                    "Received position update from",
                    current_epoch_s - self.sensed_pos_stamp_s,
                    "seconds ago",
                )
            if self.initialise_pose == True and self.sensed_pos_northings_m is not None:
                self.p_robot[0] = self.sensed_pos_northings_m
                self.p_robot[1] = self.sensed_pos_eastings_m
                self.integrated_yaw = self.integrated_yaw + self.sensed_pos_yaw_rad *self.sensed_pos_stamp_s
                self.p_robot[2] = self.integrated_yaw

                self.initialise_pose = False
                
                print("Initialised pose")

            
            if self.sensed_imu_stamp_s is not None and current_epoch_s - self.sensed_imu_stamp_s < (self.lastdt):
                # if the previous imu timestamp was known, integrate from then, otherwise use the set timestep
                if self.sensed_imu_prev_stamp_s is not None: dt = current_epoch_s - self.sensed_imu_prev_stamp_s
                else: dt=self.lastdt
                print(
                    "Received IMU update from", 
                    current_epoch_s - self.sensed_imu_stamp_s, 
                    "seconds ago",
                    )  
                # log current timestamp for calculating dt in the next loop
                self.sensed_imu_prev_stamp_s = self.sensed_imu_stamp_s

                
                self.sensed_yaw_rate = self.sensed_imu_yaw_rate_rad_s
                self.integrated_yaw = self.integrated_yaw + (self.sensed_imu_yaw_rate_rad_s * dt)


                # this handles angle wrapping between 0 and 2*pi
                self.integrated_yaw = self.integrated_yaw % (2 * np.pi)  

            if self.sensed_bottom_depth_stamp_s is not None and current_epoch_s - self.sensed_bottom_depth_stamp_s < (self.lastdt):
                print(
                    "Received Echosounder update from",
                    current_epoch_s - self.sensed_bottom_depth_stamp_s,
                    "seconds ago", 
                )                             
        
            ### NAVIGATION PROCEDURE ############################        

            ### GUIDANCE PROCEDURE ##############################

            ### CONTROL PROCEDURE ###############################
            accept_radius = 0.2
            self.trajectory.wp_progress(self.timefromstart,self.p_robot,accept_radius)
            p_ref, u_ref = self.trajectory.p_u_sample(self.timefromstart) 

            # feedback control: get pose change to desired trajectory from body
            dp = p_ref-self.p_robot
            dp[2] = (dp[2] + np.pi) % (2 * np.pi) - np.pi

            H_eb = HomogeneousTransformation(self.p_robot[0:2],self.p_robot[2])
            ds=Inverse(H_eb.H_R)@dp
            print(self.timefromstart)
            

            # update the controls 
            
            if self.lap_cpt == 0: 
                self.kn = 2*u_ref[0]/(self.L**2)
                self.kg = u_ref[0]/(self.L)
                self.lap_cpt +=1
            du = feedback_control(ds,1/self.tau_s,self.kn,self.kg)   
            u = u_ref + du
            # update control gains for the next timestep
            self.kn =  2*u[0]/(self.L**2)
            self.kg =  u[0]/(self.L)

            v = u[0]
            w = u[1]

            # determine body forces and moments (steady state)
            F_x = self.robot.k_drag*v*abs(v)
            tau_z = self.robot.B_66*w

            #calculate thruster allocation
            thrust = Inverse(self.G)@l2m([F_x,0,tau_z])        
            print("thrust ", thrust)
            #calculate thruster rpm subject to limits
            # Set thruster rate command
            self.right_rate = N2rpm(thrust[0][0])*2*np.pi/60 #rad/s
            self.left_rate = N2rpm(thrust[1][0])*2*np.pi/60#rad/s
            


            # Apply thruster command saturation limits
            if self.right_rate>200: self.right_rate=200
            if self.right_rate<-200: self.right_rate=-200
            
            if self.left_rate>200: self.left_rate=200
            if self.left_rate<-200: self.left_rate=-200


            # add the lines below
            right_N = rpm2N(self.right_rate*60/(2*np.pi))
            left_N = rpm2N(self.left_rate*60/(2*np.pi))
            print('Right Thrust, N:',right_N,', Left Thrust, N:',left_N)
            T = l2m([right_N,left_N])   
            Fb = self.G@T
            print('Surge Thrust, N:',Fb[0],', Yaw torque, Nm:',Fb[2])

                # convert thrust from b to e frame
            H_eb = HomogeneousTransformation(self.p_robot[0:2],self.p_robot[2])
            Fe = H_eb.H_R@Fb                
                
            # model dynamics e frame
            dt=self.lastdt
            self.v_robot = self.robot.model(dynamics_translation_e,dynamics_rotation_e, Fe, self.v_robot, dt)
            print('Ndot, m/s:',self.v_robot[0],'Edot, m/s:',self.v_robot[1],'Gdot, rad/s:',self.v_robot[2])

                # model kinematics from twist (b frame)
            H_eb = HomogeneousTransformation(self.p_robot[0:2],self.p_robot[2])
            vb = Inverse(H_eb.H_R)@self.v_robot

            twist = Vector(2)

            twist[0] = vb[0][0] 
            twist[1] = vb[2][0]
                        
            self.p_robot = rigid_body_kinematics(self.p_robot,twist,dt)
            self.p_robot[2] = self.p_robot[2] % (2*np.pi)

            ### SEND TO ROBOT ###########################                                 
            control_msg = Vector3()
            control_msg.x = int(self.right_rate)
            control_msg.y = int(self.left_rate)

            self.control_pub.publish(control_msg)
            print('Prop rates: R=',self.right_rate,', L=',self.left_rate,'rad/s')
            
            self.Yaw = self.p_robot[2][0] # the last [0] makes it a scalar
            self.North = self.p_robot[0][0]
            self.East = self.p_robot[1][0]
            print("north ",self.North)
            print("east ",self.East)
        ### LOG DATA ##############################
        with self.filename.open("a") as f:
            f.write(f"{current_epoch_s},{self.timefromstart},{self.right_rate},{self.left_rate},{self.lastdt},{self.Yaw},{self.North},{self.East},{self.sensed_yaw_rate},{self.integrated_yaw},{self.sensed_imu_stamp_s},{self.sensed_pos_northings_m},{self.sensed_pos_eastings_m},{self.sensed_pos_yaw_rad}, {self.sensed_pos_stamp_s}, {self.sensed_bottom_depth_stamp_s},{self.sensed_bottom_depth_m}\n")            
        
        ### VISUALISE DATA ##############################
        if self.OPERATING_MODE != 0:
            return(self.right_rate, self.left_rate, self.lastdt, self.Yaw, self.North, self.East, self.sensed_yaw_rate,self.integrated_yaw, self.sensed_imu_stamp_s, self.sensed_pos_northings_m, self.sensed_pos_eastings_m, self.sensed_pos_yaw_rad, self.sensed_pos_stamp_s, self.waypoints, self.sensed_bottom_depth_m, self.sensed_bottom_depth_stamp_s)



        ############################# END MAIN LOOP ###########################
        
def main():
    LaptopController(OPERATING_MODE = 2)
    
if __name__ == "__main__":
    main()
            