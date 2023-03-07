import typing
from scipy.special import kv # modivied bessel function of second kind of real order v
from scipy.special import kvp # derivative of modified bessel Function of second kind of real order

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

from swarmrl.models.interaction_model import Action, InteractionModel
from swarmrl.observables.subdivided_vision_cones import SubdividedVisionCones

def angle_from_vector(vec) -> float:
    return np.arctan2(vec[1], vec[0])

def vector_from_angle(angle) -> np.ndarray:
    return np.array([np.cos(angle), np.sin(angle), 0])

def get_colloids_in_vision(
    coll, other_coll, vision_half_angle=np.pi, vision_range=np.inf
) -> list:
    my_pos = np.array(coll.pos)
    my_director = coll.director
    colls_in_vision = []
    for other_p in other_coll:
        dist = other_p.pos - my_pos
        dist_norm = np.linalg.norm(dist)
        in_range = dist_norm < vision_range
        if not in_range:
            continue
        in_cone = np.arccos(np.dot(dist / dist_norm, my_director)) < vision_half_angle
        if in_cone and in_range:
            colls_in_vision.append(other_p)
    return colls_in_vision

def calc_schmell(schmellpos,colpos,diffusion_col_trans = 670): #glucose in water in mum^2/s

    '''
    #Test script
    from scipy.special import kv  # modivied bessel function of second kind of real order v
    from scipy.special import kvp  # derivative of modified bessel Function of second kind of real order
    import numpy as np
    import matplotlib.pyplot as plt
    rod_thickness = 5  # micrometer
    B = np.linspace(0.1, 10, 100)
    D = 0.0014  # translative Diffusion coefficient
    const = -2 * np.pi * D * rod_thickness / 2 * B * kvp(0, B * rod_thickness / 2)
    plt.plot(B, const)
    plt.show()

    # => the system is rather diffusion dominated if B is small and decay dominated if B is large B=np.sqrt(O/D)

    rod_thickness = 5  # micrometer
    O=0.00000    01 #  in per second
    J=0.002
    diffusion_col_transs=0.0014 # micrometer ^2 / second
    B = np.sqrt(O/diffusion_col_transs)

    const = -2 * np.pi * diffusion_col_transs * rod_thickness / 2 * B * kvp(0, B * rod_thickness / 2)
    print(const)
    A= J/const
    print(A)
    r=np.linspace(2, 200, 100)
    l = B * r
    schmell_magnitude = A*kv(0,l)
    plt.plot(r,schmell_magnitude)
    plt.show()
    '''

    delta_distance = np.linalg.norm(np.array(schmellpos) - np.array(colpos),axis=-1)
    delta_distance = np.where(delta_distance == 0, 5, delta_distance)
    # prevent zero division
    direction = (schmellpos - colpos) / np.stack([delta_distance, delta_distance], axis=-1)
    '''
    #uncomment this for gaus curves
    schmell_magnitude += np.exp(-0.5 * delta_distance ** 2 / rod_thickness ** 2)
    schmell_gradient += delta_distance * np.exp(-0.5 * delta_distance ** 2 / rod_thickness ** 2) * direction
    '''
    O=0.4 #  in per second
    J=1800 # in chemics per second whatever units the chemics are in. These values are chosen acording to the Amplitude \approx 1
    #const = -2 * np.pi * diffusion_col_trans * rod_thickness / 2 * np.sqrt(O / diffusion_col_trans) * kvp(0, np.sqrt(
    #   O / diffusion_col_trans) * rod_thickness / 2) #reuse already calculated value
    const=4182.925639571625
    # J=const*A the const sums over the chemics that flow throw an imaginary boundary at radius rod_thickness/2
    # A is the Amplitude of the potential i.e. the chemical density (google "first ficks law" for theoretical info)
    A = J / const

    l=np.sqrt(O/diffusion_col_trans)*delta_distance
    schmell_magnitude = A*kv(0,l)
    schmell_gradient = - np.stack([A* np.sqrt(O/diffusion_col_trans)*kvp(0,l),A*np.sqrt(O/diffusion_col_trans)*kvp(0,l)],axis=-1)*direction
    return schmell_magnitude, schmell_gradient

#swimm with an angle offset towards the center of the rod
class rotate_rod_angle_offset(InteractionModel):

    def __init__(
        self,
        act_force=1.0,
        act_torque=1,
        n_type=list,
        rod_center_part_id=int,
        rod_particle_type=int,
        acts_on_types: typing.List[int] = None,
    ):
        self.act_force = act_force
        self.act_torque = act_torque
        self.n_type = n_type
        self.rod_center_part_id = rod_center_part_id
        self.rod_particle_type = rod_particle_type
        if acts_on_types is None:
            acts_on_types = [0]
        self.acts_on_types = acts_on_types

    def calc_action(self, colloids) -> typing.List[Action]:
        for colloid in colloids:
            if colloid.id == self.rod_center_part_id:
                rod_center_part = colloid

        actions = []
        for colloid in colloids:

            if colloid.type not in self.acts_on_types:
                actions.append(Action())
                continue

            current_angle = angle_from_vector(colloid.director)
            to_center = rod_center_part.pos - colloid.pos
            to_center_angle = angle_from_vector(to_center)
            angle_diff = to_center_angle - current_angle-np.pi/4

            # take care of angle wraparound and bring difference to [-pi, pi]
            if angle_diff >= np.pi:
                angle_diff -= 2 * np.pi
            if angle_diff <= -np.pi:
                angle_diff += 2 * np.pi
            torque_z = np.sin(angle_diff) * self.act_torque

            actions.append(
                Action(force=self.act_force, torque=np.array([0, 0, torque_z]))
            )

        return actions

#swimm to rod because the rod has a schmell/potential and swimm to it with a angle offset
class rotate_rod_schmell(InteractionModel):
    def __init__(
        self,
        act_force=1.0,
        act_torque=1,
        n_type=[],
        rod_center_part_id=42,
        rod_particle_type=4,
        rod_thickness=3,
        acts_on_types: typing.List[int] = None,
    ):
        self.act_force = act_force
        self.act_torque = act_torque
        self.n_type = n_type
        self.rod_center_part_id = rod_center_part_id
        self.rod_particle_type = rod_particle_type
        self.rod_thickness = rod_thickness
        if acts_on_types is None:
            acts_on_types = [0]
        self.acts_on_types = acts_on_types

    def calc_action(self, colloids) -> typing.List[Action]:
        actions = []
        #fig, ax = plt.subplots(1,1)
        for colloid in colloids:
            if colloid.type not in self.acts_on_types:
                actions.append(Action())
                continue
            schmell_magnitude=0
            schmell_gradient=[0,0,0]
            for rod_colloid in colloids:
                if rod_colloid.type == self.rod_particle_type:
                    delta_distance = np.linalg.norm(rod_colloid.pos-colloid.pos)
                    direction = (rod_colloid.pos-colloid.pos)/delta_distance
                    schmell_magnitude += np.exp(-0.5*delta_distance**2/self.rod_thickness.magnitude**2)
                    schmell_gradient += delta_distance*np.exp(-0.5*delta_distance**2/self.rod_thickness.magnitude**2)*direction

            schmell_magnitude=schmell_magnitude/self.n_type[self.rod_particle_type]

            current_angle = angle_from_vector(colloid.director)
            to_center_angle = angle_from_vector(schmell_gradient)

            angle_diff = to_center_angle - current_angle - np.pi/4 ###########

            # take care of angle wraparound and bring difference to [-pi, pi]
            if angle_diff >= np.pi:
                angle_diff -= 2 * np.pi
            if angle_diff <= -np.pi:
                angle_diff += 2 * np.pi
            torque_z = np.sin(angle_diff) * self.act_torque

            actions.append(
                Action(force=self.act_force, torque=np.array([0, 0, torque_z]))
            )

            #x = colloid.pos[0]
            #y = colloid.pos[1]
            #dx = 10*schmell_gradient[0]/np.linalg.norm(schmell_gradient)
            #dy = 10*schmell_gradient[1]/np.linalg.norm(schmell_gradient)
            #ax.arrow(x,y,dx,dy,width=1)
        #plt.show()
        return actions

#only swimm forward if you face with a certain angle range towards the center
class rotate_rod_rodeo(InteractionModel):

    def __init__(
        self,
        act_force=1.0,
        act_torque=1,
        n_type=list,
        rod_center_part_id=int,
        rod_particle_type=int,
        acts_on_types: typing.List[int] = None,
    ):
        self.act_force = act_force
        self.act_torque = act_torque
        self.n_type = n_type
        self.rod_center_part_id = rod_center_part_id
        self.rod_particle_type = rod_particle_type
        if acts_on_types is None:
            acts_on_types = [0]
        self.acts_on_types = acts_on_types

    def calc_action(self, colloids) -> typing.List[Action]:
        for colloid in colloids:
            if colloid.id == self.rod_center_part_id:
                rod_center_part = colloid

        actions = []
        for colloid in colloids:
            if colloid.type not in self.acts_on_types:
                actions.append(Action())
                continue

            current_angle = angle_from_vector(colloid.director)
            to_center = rod_center_part.pos - colloid.pos
            to_center_angle = angle_from_vector(to_center)
            angle_diff = to_center_angle - current_angle
            if angle_diff>0 and angle_diff<np.pi/2:
                actions.append(
                    Action(force=self.act_force)
                )
            else:
                actions.append(Action())

        return actions

#swimm forward if you face with a certain angle range towards the center based on schmell
class rotate_rod_schmell_rodeo(InteractionModel):
    def __init__(
        self,
        act_force=1.0,
        act_torque=1,
        n_type=[],
        rod_center_part_id=42,
        rod_particle_type=4,
        rod_thickness=3,
        acts_on_types: typing.List[int] = None,
    ):
        self.act_force = act_force
        self.act_torque = act_torque
        self.n_type = n_type
        self.rod_center_part_id = rod_center_part_id
        self.rod_particle_type = rod_particle_type
        self.rod_thickness = rod_thickness
        if acts_on_types is None:
            acts_on_types = [0]
        self.acts_on_types = acts_on_types

    def calc_action(self, colloids) -> typing.List[Action]:
        actions = []
        #fig, ax = plt.subplots(1,1)
        for colloid in colloids:
            if colloid.type not in self.acts_on_types:
                actions.append(Action())
                continue
            schmell_magnitude=0
            schmell_gradient=[0,0,0]
            for rod_colloid in colloids:
                if rod_colloid.type == self.rod_particle_type:
                    delta_distance = np.linalg.norm(rod_colloid.pos-colloid.pos)
                    direction = (rod_colloid.pos-colloid.pos)/delta_distance
                    schmell_magnitude += np.exp(-0.5*delta_distance**2/self.rod_thickness.magnitude**2)
                    schmell_gradient += delta_distance*np.exp(-0.5*delta_distance**2/self.rod_thickness.magnitude**2)*direction

            schmell_magnitude=schmell_magnitude/self.n_type[self.rod_particle_type]

            current_angle = angle_from_vector(colloid.director)
            to_center_angle = angle_from_vector(schmell_gradient)

            angle_diff = to_center_angle - current_angle

            if angle_diff > 0 and angle_diff < np.pi / 2:
                actions.append(
                    Action(force=self.act_force)
                )
            else:
                actions.append(Action())

            #x = colloid.pos[0]
            #y = colloid.pos[1]
            #dx = 10*schmell_gradient[0]/np.linalg.norm(schmell_gradient)
            #dy = 10*schmell_gradient[1]/np.linalg.norm(schmell_gradient)
            #ax.arrow(x,y,dx,dy,width=1)
        #plt.show()
        return actions

class rotate_rod_border_schmell_symmetric(InteractionModel):
    def __init__(
        self,
        act_force=1.0,
        act_torque=1,
        n_type=[],
        rod_schmell_part_id=[42,43],
        rod_center_part_id=42,
        rod_particle_type=4,
        rod_thickness=3,
        radius_colloid=3,
        force_team_spirit_fac=0,
        diffusion_col_trans=0,
        rod_break_ang_vel= 0.0042,
        rod_break=False,
        acts_on_types: typing.List[int] = None,
    ):
        self.act_force = act_force
        self.act_torque = act_torque
        self.n_type = n_type
        self.rod_schmell_part_id = rod_schmell_part_id
        self.rod_center_part_id = rod_center_part_id
        self.rod_particle_type = rod_particle_type
        self.rod_thickness = rod_thickness
        self.radius_colloid = radius_colloid
        self.force_team_spirit_fac = force_team_spirit_fac
        self.diffusion_col_trans=diffusion_col_trans
        self.rod_break_ang_vel = rod_break_ang_vel
        self.rod_break = rod_break
        self.step_counter = 0
        if acts_on_types is None:
            acts_on_types = [0]
        self.acts_on_types = acts_on_types
        self.rod_break_force = 250.0
        self.break_angle = np.pi/64
        self.break_max_angle = np.pi/4
        self.turn_direction = 1


    def calc_action(self, colloids) -> typing.List[Action]:
        actions = []


        for colloid in colloids:
            if colloid.id == self.rod_center_part_id and self.rod_break is True:
                self.step_counter += 1 * self.turn_direction
                print(self.step_counter)
                current_angle = angle_from_vector(colloid.director)
                to_center_angle = self.step_counter * self.rod_break_ang_vel
                angle_diff = to_center_angle - current_angle

                # take care of angle wraparound and bring difference to [-pi, pi]
                if angle_diff >= np.pi:
                    angle_diff -= 2 * np.pi
                if angle_diff <= -np.pi:
                    angle_diff += 2 * np.pi

                if angle_diff > self.break_max_angle:
                    torque_z_center = self.act_torque * self.rod_break_force * (2 * (angle_diff-self.break_max_angle) +(self.break_max_angle-self.break_angle))
                    print("emergency break for rod activated+")
                elif angle_diff < self.break_max_angle and angle_diff > self.break_angle:
                    torque_z_center = self.act_torque * self.rod_break_force * (angle_diff-self.break_angle)
                elif angle_diff < self.break_angle and angle_diff > 0:
                    torque_z_center = 0
                    self.turn_direction = -1
                elif angle_diff > -self.break_angle and angle_diff < 0:
                    torque_z_center = 0
                    self.turn_direction = 1
                elif angle_diff > -self.break_max_angle and angle_diff < -self.break_angle:
                    torque_z_center = self.act_torque * self.rod_break_force  * (angle_diff+self.break_angle)
                elif angle_diff < -self.break_max_angle:
                    torque_z_center = self.act_torque * self.rod_break_force *( 2 * (angle_diff+self.break_angle) + (-self.break_max_angle+self.break_angle))
                    print("emergency break for rod activated-")
                else:
                    print("something strange happend to the if statement in the rod Break")

                actions.append(
                    Action(force=0, torque=np.array([0, 0, torque_z_center]))
                )
            elif colloid.type not in self.acts_on_types:
                actions.append(Action())
                continue

            schmell_magnitude=0
            schmell_gradient=[0,0]
            for rod_colloid in colloids:
                if len(self.rod_schmell_part_id) != 2:
                    print("smellsourcecounter",len(self.rod_schmell_part_id))
                if rod_colloid.id in self.rod_schmell_part_id:
                    mag,grad=calc_schmell(rod_colloid.pos[:2], colloid.pos[:2])
                    schmell_magnitude += mag
                    schmell_gradient += grad
            n_neighbor = 0
            '''
            for fellow_colloid in colloids:
                if fellow_colloid.type == 0 and fellow_colloid.id!=colloid.id:
                    delta_distance = np.linalg.norm(fellow_colloid.pos - colloid.pos)
                    if delta_distance < self.radius_colloid.magnitude*3:
                        n_neighbor+=1
            #This calculation tries to keep the average act_force constant despite different spreading due to force_mult
            #the negative value is the average value of neighbors in a trial simulation ->analysis_single.py ->mean_neighbors
            force_mult=1+(-0.1627+n_neighbor)*self.force_team_spirit_fac
            '''

            current_angle = angle_from_vector(colloid.director)
            to_center_angle = angle_from_vector(schmell_gradient)

            angle_diff = to_center_angle - current_angle

            # take care of angle wraparound and bring difference to [-pi, pi]
            if angle_diff >= np.pi:
                angle_diff -= 2 * np.pi
            if angle_diff <= -np.pi:
                angle_diff += 2 * np.pi
            torque_z = np.arctan(angle_diff) * self.act_torque

            actions.append(
                Action(force=self.act_force, torque=np.array([0, 0, torque_z]))
            )

        return actions


# this actions does not turn the rod because if the colloid swimm against the rod they do not get al larger schmell
# therefore they think they move in the wrong direction and therefore they stop moving the rod.
# (To get around this the colloids shall not turn when they touch the rod. not approved)
class rotate_rod_border_schmell_symmetric_gradient_memory(InteractionModel):
    def __init__(
        self,
        act_force=1.0,
        act_torque=1,
        n_type=[],
        rod_border_parts_id=[42,43],
        rod_particle_type=4,
        rod_thickness=3,
        radius_colloid=3,
        force_team_spirit_fac=0,
        acts_on_types: typing.List[int] = None,
    ):
        self.act_force = act_force
        self.act_torque = act_torque
        self.n_type = n_type
        self.schmell_magnitude_memory=np.zeros(np.sum(self.n_type))
        self.rod_border_parts_id = rod_border_parts_id
        self.rod_particle_type = rod_particle_type
        self.rod_thickness = rod_thickness
        self.radius_colloid=radius_colloid
        self.force_team_spirit_fac=force_team_spirit_fac
        if acts_on_types is None:
            acts_on_types = [0]
        self.acts_on_types = acts_on_types
        self.counter=0
        self.run_vs_steer=np.zeros(np.sum(self.n_type))
        self.touch=0

    def calc_action(self, colloids) -> typing.List[Action]:
        actions = []
        #fig, ax = plt.subplots(1,1)
        self.counter+=1



        for colloid in colloids:
            if colloid.type not in self.acts_on_types:
                actions.append(Action())
                continue
            schmell_magnitude=0
            schmell_gradient=[0,0,0]
            for rod_colloid in colloids:
                if len(self.rod_border_parts_id)!=2:
                    print("smellsourcecounter",len(self.rod_border_parts_id))
                if rod_colloid.id in self.rod_border_parts_id:
                    delta_distance = np.linalg.norm(rod_colloid.pos[:2]-colloid.pos[:2]) # die z richtung glitch herum deshalb wird sie hier rausgelassen
                    if self.rod_thickness.magnitude * 100 < delta_distance:
                        print("rod pos",rod_colloid.pos,"colloid pos",colloid.pos)
                    schmell_magnitude += np.exp(-0.5*delta_distance**2/self.rod_thickness.magnitude**2)
            n_neighbor = 0
            for fellow_colloid in colloids:
                if fellow_colloid.type == 0 and fellow_colloid.id!=colloid.id:
                    delta_distance = np.linalg.norm(fellow_colloid.pos[:2] - colloid.pos[:2])
                    if delta_distance < self.radius_colloid.magnitude*3:
                        n_neighbor+=1
            #This calculation tries to keep the average act_force constant despite different spreading due to force_mult
            #the negative value is the average value of neighbors in a trial simulation ->analysis_single.py ->mean_neighbors
            force_mult = 1+(-0.1627+n_neighbor)*self.force_team_spirit_fac

            schmell_magnitude = schmell_magnitude/2
            # does the rod and the colloid touch, if so then the colloid shall not steer
            self.touch=0
            for rod_colloid in colloids:
                if rod_colloid.type not in self.acts_on_types:
                    delta_distance = np.linalg.norm(rod_colloid.pos[:2]-colloid.pos[:2])
                    if delta_distance < (self.radius_colloid.magnitude+self.rod_thickness.magnitude/2)*2.5:
                        self.touch=1
                    
            if self.counter % 100 == 0:
                if self.schmell_magnitude_memory[colloid.id] > schmell_magnitude and self.run_vs_steer[colloid.id]!=1 and self.touch!=1:
                    self.run_vs_steer[colloid.id] = 1
                else:
                    self.run_vs_steer[colloid.id] = 0
                self.schmell_magnitude_memory[colloid.id] = schmell_magnitude

            if  self.run_vs_steer[colloid.id] == 1:
                torque_z = (-1)**colloid.id*self.act_torque # every odd colloid steers permanently right ...
                force_mult = 0
            else:
                torque_z = 0



            actions.append(
                Action(force=self.act_force*force_mult, torque=np.array([0, 0, torque_z]))
            )

            #x = colloid.pos[0]
            #y = colloid.pos[1]
            #dx = 10*schmell_gradient[0]/np.linalg.norm(schmell_gradient)
            #dy = 10*schmell_gradient[1]/np.linalg.norm(schmell_gradient)
            #ax.arrow(x,y,dx,dy,width=1)
        #plt.show()
        return actions


class rotate_rod_center_schmell_gradient_memory(InteractionModel):
    def __init__(
            self,
            act_force=42,
            act_torque=42,
            n_type=[],
            rod_schmell_part_id=[42],
            rod_particle_type=42,
            rod_thickness=42,
            radius_colloid=42,
            force_team_spirit_fac=0,
            phase_len=[4,4],
            acts_on_types: typing.List[int] = None,
    ):
        self.act_force = act_force
        self.act_torque = act_torque
        self.n_type = n_type
        self.rod_schmell_part_id = rod_schmell_part_id
        self.rod_particle_type = rod_particle_type
        self.rod_thickness = rod_thickness
        self.radius_colloid = radius_colloid
        self.force_team_spirit_fac = force_team_spirit_fac
        self.phase_len = phase_len
        if acts_on_types is None:
            acts_on_types = [0]
        self.acts_on_types = acts_on_types

        self.n_col = sum(self.n_type)
        self.n_phases = 2  # run, steer
        self.n_memory = 2  # schmellmemory
        self.col_RAM = np.zeros((self.n_col, self.n_phases + self.n_memory))
        self.col_RAM[:,0] = 1 # starting in phase 0 with a single step

        #steer_angle=36  # degree
        #steer_steps=int(np.ceil(steer_angle/0.8)) # 0.8 degree/step
        #run_distance=4 # mu
        #run_steps = int(np.ceil(run_distance/0.1)) # mu per step
        #self.phase_len = [run_steps,steer_steps] #run ,steer


        self.phase_trans = [1,0]


        #print(self.col_RAM[1])

    def calc_action(self, colloids) -> typing.List[Action]:
        actions = []

        for colloid in colloids:
            if colloid.type not in self.acts_on_types:
                actions.append(Action())
                continue

            #setup phases
            if sum(self.col_RAM[colloid.id, :self.n_phases]) == 0:
                phase = 0
            else:
                #print(self.col_RAM[colloid.id, :self.n_phases])
                [phase], = np.where(self.col_RAM[colloid.id, :self.n_phases] != 0)

            schmell_magnitude = 0
            for rod_colloid in colloids:
                if rod_colloid.id in self.rod_schmell_part_id:
                    a,_= calc_schmell(rod_colloid.pos[:2],colloid.pos[:2]) # die z richtung glitch herum deshalb wird sie hier rausgelassen
                    schmell_magnitude += a

            if phase == 0 and self.col_RAM[colloid.id, phase] == self.phase_len[phase]:
                if self.col_RAM[colloid.id,self.n_phases] < schmell_magnitude:  #self.N_phase is the index vor schmellmemory
                    self.col_RAM[colloid.id, phase]=0
                    phase=0  # jump to run phase and head straight forward
                    self.col_RAM[colloid.id, phase] = 1
                else:
                    self.col_RAM[colloid.id, phase] = 0
                    phase = 1  # jump to steer phase and head straight forward
                    self.col_RAM[colloid.id, phase] = 1
                    #pass
                self.col_RAM[colloid.id,self.n_phases] = schmell_magnitude

            if phase==0: #run
                force_mult = 1
                torque_z =  0
            elif phase==1: #steer
                force_mult = 1
                torque_z =  1
            else:
                raise Exception("Colloid doesn't now what to do. Unexpected phase identifier selected")

            actions.append(
                Action(force=self.act_force * force_mult, torque=np.array([0, 0, self.act_torque* torque_z]))
            )

            #propagate phases
            for j in range(self.n_phases):
                if phase == j and self.col_RAM[colloid.id, phase] == self.phase_len[j]:
                    self.col_RAM[colloid.id, phase] = 0
                    phase = self.phase_trans[j]

            self.col_RAM[colloid.id, phase] += 1

        #print(self.col_RAM[1])

        return actions


class rotate_rod_center_schmell_gradient_memory_touch(InteractionModel):
    def __init__(
            self,
            act_force=42,
            act_torque=42,
            n_type=[],
            rod_schmell_part_id=[42],
            rod_particle_type=42,
            rod_thickness=42,
            radius_colloid=42,
            force_team_spirit_fac=0,
            phase_len=[42,42],
            acts_on_types: typing.List[int] = None,
    ):
        self.act_force = act_force
        self.act_torque = act_torque
        self.n_type = n_type
        self.rod_schmell_part_id = rod_schmell_part_id
        self.rod_particle_type = rod_particle_type
        self.rod_thickness = rod_thickness
        self.radius_colloid = radius_colloid
        self.force_team_spirit_fac = force_team_spirit_fac
        self.phase_len = phase_len
        if acts_on_types is None:
            acts_on_types = [0]
        self.acts_on_types = acts_on_types

        self.n_col = sum(self.n_type)
        self.n_phases = 2  # run, steer
        self.n_memory = 2  # schmellmemory
        self.col_RAM = np.zeros((self.n_col, self.n_phases + self.n_memory))
        self.col_RAM[:,0] = 1 # starting in phase 0 with a single step

        #steer_angle=36  # degree
        #steer_steps=int(np.ceil(steer_angle/0.8)) # 0.8 degree/step
        #run_distance=4 # mu
        #run_steps = int(np.ceil(run_distance/0.1)) # mu per step
        #self.phase_len = [run_steps,steer_steps] #run ,steer


        self.phase_trans = [1,0]

        #print(self.col_RAM[1])

    def calc_action(self, colloids) -> typing.List[Action]:
        actions = []
        # fig, ax = plt.subplots(1,1)


        for colloid in colloids:
            if colloid.type not in self.acts_on_types:
                actions.append(Action())
                continue


            #setup phases
            if sum(self.col_RAM[colloid.id, :self.n_phases]) == 0:
                phase = 0
            else:
                #print(self.col_RAM[colloid.id, :self.n_phases])
                [phase], = np.where(self.col_RAM[colloid.id, :self.n_phases] != 0)

            '''
            n_neighbor = 0
            for fellow_colloid in colloids:
                if fellow_colloid.type == 0 and fellow_colloid.id != colloid.id:
                    delta_distance = np.linalg.norm(fellow_colloid.pos[:2] - colloid.pos[:2])
                    if delta_distance < self.radius_colloid.magnitude * 3:
                        n_neighbor += 1
            # This calculation tries to keep the average act_force constant despite different spreading due to force_mult
            # the negative value is the average value of neighbors in a trial simulation ->analysis_single.py ->mean_neighbors
            force_mult = 1 + (-0.1627 + n_neighbor) * self.force_team_spirit_fac
            '''


            # does the rod and the colloid touch, if so then the colloid shall not steer

            self.touch = 0
            for rod_colloid in colloids:
                if rod_colloid.type not in self.acts_on_types:
                    delta_distance = np.linalg.norm(rod_colloid.pos[:2] - colloid.pos[:2])
                    if delta_distance < (self.radius_colloid.magnitude + self.rod_thickness.magnitude / 2) * 2.5:
                        phase = 0
                        self.col_RAM[colloid.id,:self.n_phases] = self.col_RAM[colloid.id,:self.n_phases] * 0
                        self.col_RAM[colloid.id,phase] = self.phase_len[0]-1


            schmell_magnitude = 0
            for rod_colloid in colloids:
                if rod_colloid.id in self.rod_schmell_part_id:
                    a,_= calc_schmell(rod_colloid.pos[:2],colloid.pos[:2]) # die z richtung glitch herum deshalb wird sie hier rausgelassen
                    schmell_magnitude += a

            if phase==1 and self.col_RAM[colloid.id,phase]==1: # if the first steer step is about to come
                if self.col_RAM[colloid.id,self.n_phases] < schmell_magnitude:  #self.N_phase is the index vor schmellmemory
                    self.col_RAM[colloid.id, phase]=0
                    phase=0  # jump to run phase and head straight forward
                    self.col_RAM[colloid.id, phase] = 1
                else:
                    pass # stay in the steer phase
                self.col_RAM[colloid.id,self.n_phases] = schmell_magnitude

            if phase==0: #run
                force_mult = 1
                torque_z =  0
            elif phase==1: #steer
                force_mult = 0
                torque_z =  1
            else:
                raise Exception("Colloid doesn't now what to do. Unexpected phase identifier selected")

            actions.append(
                Action(force=self.act_force * force_mult, torque=np.array([0, 0, self.act_torque* torque_z]))
            )

            #propagate phases
            for j in range(self.n_phases):
                if phase == j and self.col_RAM[colloid.id, phase] == self.phase_len[j]:
                    self.col_RAM[colloid.id, phase] = 0
                    phase = self.phase_trans[j]

            self.col_RAM[colloid.id, phase] += 1

        #print(self.col_RAM[1])

        return actions


class rotate_rod_phase_vorlage(InteractionModel):
    def __init__(
            self,
            act_force=42,
            act_torque=42,
            n_type=[],
            rod_schmell_part_id=[42],
            rod_particle_type=42,
            rod_thickness=42,
            radius_colloid=42,
            force_team_spirit_fac=0,
            phase_len=[42,42],
            acts_on_types: typing.List[int] = None,
    ):
        self.act_force = act_force
        self.act_torque = act_torque
        self.n_type = n_type
        self.rod_schmell_part_id = rod_schmell_part_id
        self.rod_particle_type = rod_particle_type
        self.rod_thickness = rod_thickness
        self.radius_colloid = radius_colloid
        self.force_team_spirit_fac = force_team_spirit_fac
        self.phase_len = phase_len
        if acts_on_types is None:
            acts_on_types = [0]
        self.acts_on_types = acts_on_types

        self.n_col = sum(self.n_type)
        self.n_phases = 2  # run, steer
        self.n_memory = 2  # schmellmemory
        self.col_RAM = np.zeros((self.n_col, self.n_phases + self.n_memory))
        self.col_RAM[:,0] = 1 # starting in phase 0 with a single step

        #steer_angle=36  # degree
        #steer_steps=int(np.ceil(steer_angle/0.8)) # 0.8 degree/step
        #run_distance=4 # mu
        #run_steps = int(np.ceil(run_distance/0.1)) # mu per step
        #self.phase_len = [run_steps,steer_steps] #run ,steer


        self.phase_trans = [1,0]

        #print(self.col_RAM[1])
    def calc_action(self, colloids) -> typing.List[Action]:
        actions = []
        # fig, ax = plt.subplots(1,1)


        for colloid in colloids:
            if colloid.type not in self.acts_on_types:
                actions.append(Action())
                continue


            #setup phases
            if sum(self.col_RAM[colloid.id, :self.n_phases]) == 0:
                phase = 0
            else:
                #print(self.col_RAM[colloid.id, :self.n_phases])
                [phase], = np.where(self.col_RAM[colloid.id, :self.n_phases] != 0)

            '''
            n_neighbor = 0
            for fellow_colloid in colloids:
                if fellow_colloid.type == 0 and fellow_colloid.id != colloid.id:
                    delta_distance = np.linalg.norm(fellow_colloid.pos[:2] - colloid.pos[:2])
                    if delta_distance < self.radius_colloid.magnitude * 3:
                        n_neighbor += 1
            # This calculation tries to keep the average act_force constant despite different spreading due to force_mult
            # the negative value is the average value of neighbors in a trial simulation ->analysis_single.py ->mean_neighbors
            force_mult = 1 + (-0.1627 + n_neighbor) * self.force_team_spirit_fac
            '''


            # does the rod and the colloid touch, if so then the colloid shall not steer

            self.touch = 0
            for rod_colloid in colloids:
                if rod_colloid.type not in self.acts_on_types:
                    delta_distance = np.linalg.norm(rod_colloid.pos[:2] - colloid.pos[:2])
                    if delta_distance < (self.radius_colloid.magnitude + self.rod_thickness.magnitude / 2) * 2.5:
                        phase = 0
                        self.col_RAM[colloid.id,:self.n_phases] = self.col_RAM[colloid.id,:self.n_phases] * 0
                        self.col_RAM[colloid.id,phase] = self.phase_len[0]-1


            schmell_magnitude = 0
            for rod_colloid in colloids:
                if rod_colloid.id in self.rod_schmell_part_id:
                    a,_= calc_schmell(rod_colloid.pos[:2],colloid.pos[:2]) # die z richtung glitch herum deshalb wird sie hier rausgelassen
                    schmell_magnitude += a

            if phase==1 and self.col_RAM[colloid.id,phase]==1: # if the first steer step is about to come
                if self.col_RAM[colloid.id,self.n_phases] < schmell_magnitude:  #self.N_phase is the index vor schmellmemory
                    self.col_RAM[colloid.id, phase]=0
                    phase=0  # jump to run phase and head straight forward
                    self.col_RAM[colloid.id, phase] = 1
                else:
                    pass # stay in the steer phase
                self.col_RAM[colloid.id,self.n_phases] = schmell_magnitude

            if phase==0: #run
                force_mult = 1
                torque_z =  0
            elif phase==1: #steer
                force_mult = 0
                torque_z =  1
            else:
                raise Exception("Colloid doesn't know what to do. Unexpected phase identifier selected")

            actions.append(
                Action(force=self.act_force * force_mult, torque=np.array([0, 0, self.act_torque * torque_z]))
            )

            #propagate phases
            for j in range(self.n_phases):
                if phase == j and self.col_RAM[colloid.id, phase] == self.phase_len[j]:
                    self.col_RAM[colloid.id, phase] = 0
                    phase = self.phase_trans[j]

            self.col_RAM[colloid.id, phase] += 1

        #print(self.col_RAM[1])

        return actions

class run_straight(InteractionModel):
    def __init__(self,act_force=42):
        n=42
        self.act_force=act_force

    def calc_action(self, colloids) -> typing.List[Action]:
        actions = []
        for colloid in colloids:
            actions.append(Action(force=self.act_force *10, torque=np.array([0, 0, 0])))
        return actions

class do_nothing(InteractionModel):
    def __init__(self):
        n=42

    def calc_action(self, colloids) -> typing.List[Action]:
        actions = []
        for colloid in colloids:
            actions.append(Action())
        return actions


class zickzack_pointfind(InteractionModel):
    def __init__(
            self,
            act_force=42,
            act_torque=42,
            n_type=[],
            rod_schmell_part_id=[42],
            phase_len=[30, 420, 420, 64],
            diffusion_coeff=42,
            zick_angle = 20,
            t_step = 0.2,
            steer_speed = 0.8 * np.pi / 180,  # rad per step
            run_speed = 0.1, #mu/step
            len_run =20 ,# mu
            experiment_engine = False,
            acts_on_types: typing.List[int] = None,
    ):
        self.act_force = act_force
        self.act_torque = act_torque
        self.n_type = n_type
        self.rod_schmell_part_id = rod_schmell_part_id

        self.steer_speed = steer_speed # 0.8 * np.pi / 180  # rad per step
        self.len_run = len_run  # mu

        self.diffusion_coeff = diffusion_coeff # mu ^2 per second transversal
        self.t_step = t_step #second
        self.zick_angle = zick_angle * np.pi / 180
        self.written_info_data = []


        # steer_angle=36*np.pi/180   # 36 degree
        # steer_steps=int(np.ceil(steer_angle/self.steer_speed)) # self.steer_speed rad/step
        # run_distance=4 # mu
        # run_steps = int(np.ceil(run_distance/0.1)) # mu per step
        # self.phase_len = [run_steps,steer_steps] #run ,steer

        self.experiment_engine = experiment_engine
        if acts_on_types is None:
            acts_on_types = [0]
        self.acts_on_types = acts_on_types

        self.n_col = sum(self.n_type)
        self.n_phases = 4  # run, steer_left, steer_right, observe_think,
        self.n_memory = 9  # measure_count, schmellpartialsum, last schnellmagnitude, schelldifferenzpartialsum, last schmell potential, last schmell potential gradient,old_angle,angle_steer, start up counter
        self.col_RAM = np.zeros((self.n_col, self.n_phases + self.n_memory))
        self.phase_len = np.ones((self.n_col, len(phase_len)))
        self.phase_len[:,0] *= int(self.len_run / run_speed ) 
        self.phase_len[:,1] *= phase_len[1]
        self.phase_len[:, 2] *= phase_len[2]
        self.phase_len[:, 3] *= phase_len[3]
        self.col_RAM[:, 3] = 1  # starting in phase 0 with a single step

        self.i_run = 0
        self.i_steer_left = 1
        self.i_steer_right = 2
        self.i_observe_think = 3
        self.i_measure_count = 4
        self.i_schmell_psum = 5
        self.i_old_schmell_magnitude = 6
        self.i_schmell_diff_psum = 7
        self.i_old_schmell_pot = 8
        self.i_old_schmell_pot_grad = 9
        self.i_old_angle = 10
        self.i_angle_steer = 11
        self.i_start_up = 12
        self.index = 4

        self.phase_trans = [3, 3, 3, 0]

        # print(self.col_RAM[1])

    def gather_measurement(self, c_id, schmell_magnitude):
        self.col_RAM[c_id, self.i_schmell_diff_psum] += abs(schmell_magnitude - self.col_RAM[c_id, self.i_old_schmell_magnitude])
        self.col_RAM[c_id, self.i_schmell_psum] += schmell_magnitude
        self.col_RAM[c_id, self.i_measure_count] += 1

    def calc_pot_gradpot(self, c_id , c_pos, c_rod_pos):
        schmell_pot = self.col_RAM[c_id, self.i_schmell_psum] / (self.col_RAM[c_id, self.i_measure_count]+1) # There is one more measurement done here
        mystical_factor=5.5
        schmell_pot_grad = self.col_RAM[c_id, self.i_schmell_diff_psum]*mystical_factor / (self.col_RAM[c_id, self.i_measure_count]*np.sqrt(4 * self.diffusion_coeff * self.t_step / np.pi ))
        schmell_magnitude = 0
        schmell_grad = 0
        a, b = calc_schmell(c_rod_pos[:2], c_pos[:2])  # die z richtung glitch herum deshalb wird sie hier rausgelassen
        schmell_magnitude = a
        schmell_grad = np.linalg.norm(b)
        if c_id == self.index :
            print(c_pos)
            print("pot", schmell_pot, "ana pot", schmell_magnitude ,"frac",schmell_pot/schmell_magnitude)
            print("grad", schmell_pot_grad, "ana_pot_grad", schmell_grad, "frac",schmell_pot_grad/schmell_grad)

        return schmell_pot, schmell_pot_grad

    def resolve_sign(self, c_id, angle_new):
        index=self.index
        a = abs((1) * abs(self.col_RAM[c_id, self.i_old_angle]) - self.col_RAM[c_id, self.i_angle_steer] - (1) * angle_new)
        b = abs((1) * abs(self.col_RAM[c_id, self.i_old_angle]) - self.col_RAM[c_id, self.i_angle_steer] - (-1) * angle_new)
        c = abs((-1) * abs(self.col_RAM[c_id, self.i_old_angle]) - self.col_RAM[c_id, self.i_angle_steer] - (1) * angle_new)
        d = abs((-1) * abs(self.col_RAM[c_id, self.i_old_angle]) - self.col_RAM[c_id, self.i_angle_steer] - (-1) * angle_new)

        e = abs((1) * abs(self.col_RAM[c_id, self.i_old_angle]) - self.col_RAM[c_id, self.i_angle_steer] - (1) * angle_new+360)
        f = abs((1) * abs(self.col_RAM[c_id, self.i_old_angle]) - self.col_RAM[c_id, self.i_angle_steer] - (-1) * angle_new+360)
        g = abs((-1) * abs(self.col_RAM[c_id, self.i_old_angle]) - self.col_RAM[c_id, self.i_angle_steer] - (1) * angle_new+360)
        h = abs((-1) * abs(self.col_RAM[c_id, self.i_old_angle]) - self.col_RAM[c_id, self.i_angle_steer] - (-1) * angle_new+360)

        i = abs((1) * abs(self.col_RAM[c_id, self.i_old_angle]) - self.col_RAM[c_id, self.i_angle_steer] - (1) * angle_new-360)
        j = abs((1) * abs(self.col_RAM[c_id, self.i_old_angle]) - self.col_RAM[c_id, self.i_angle_steer] - (-1) * angle_new-360)
        k = abs((-1) * abs(self.col_RAM[c_id, self.i_old_angle]) - self.col_RAM[c_id, self.i_angle_steer] - (1) * angle_new-360)
        l = abs((-1) * abs(self.col_RAM[c_id, self.i_old_angle]) - self.col_RAM[c_id, self.i_angle_steer] - (-1) * angle_new-360)

        deg=180/np.pi
        if c_id == index:
            print(str(round(a*deg)) + "= a=|(1)  *" + str(round(abs(self.col_RAM[c_id, self.i_old_angle])*deg)) + "-(" + str(round(self.col_RAM[c_id, self.i_angle_steer]*deg)) + ")- (1 ) *" + str(round(angle_new*deg)) + "|")
            print(str(round(b*deg)) + "= b=|(1)  *" + str(round(abs(self.col_RAM[c_id, self.i_old_angle])*deg)) + "-(" + str(round(self.col_RAM[c_id, self.i_angle_steer]*deg)) + ")- (-1) *" + str(round(angle_new*deg)) + "|")
            print(str(round(c*deg)) + "= c=|(-1) *" + str(round(abs(self.col_RAM[c_id, self.i_old_angle])*deg)) + "-(" + str(round(self.col_RAM[c_id, self.i_angle_steer]*deg)) + ")- (1 ) *" + str(round(angle_new*deg)) + "|")
            print(str(round(d*deg)) + "= d=|(-1) *" + str(round(abs(self.col_RAM[c_id, self.i_old_angle])*deg)) + "-(" + str(round(self.col_RAM[c_id, self.i_angle_steer]*deg)) + ")- (-1) *" + str(round(angle_new*deg)) + "|")

        if a == np.min([a, b, c, d, e,f,g,h,i,j,k,l]) or e == np.min([a, b, c, d, e,f,g,h,i,j,k,l]) or i == np.min([a, b, c, d, e,f,g,h,i,j,k,l]):
            #pass
            angle_new=angle_new
            self.col_RAM[c_id,self.i_old_angle]=self.col_RAM[c_id,self.i_old_angle]
            if c_id == index:
                print("a,e,i", angle_new*deg, self.col_RAM[c_id,self.i_old_angle]*deg)
        elif b == np.min([a, b, c, d, e,f,g,h,i,j,k,l]) or f == np.min([a, b, c, d, e,f,g,h,i,j,k,l]) or j == np.min([a, b, c, d, e,f,g,h,i,j,k,l]):
            angle_new = -angle_new
            self.col_RAM[c_id,self.i_old_angle]=self.col_RAM[c_id,self.i_old_angle]
            if c_id == index:
                print("b,f,j", angle_new*deg, self.col_RAM[c_id,self.i_old_angle]*deg)
        elif c == np.min([a, b, c, d, e,f,g,h,i,j,k,l]) or g == np.min([a, b, c, d, e,f,g,h,i,j,k,l]) or k == np.min([a, b, c, d, e,f,g,h,i,j,k,l]):  # should not be called because the sign of the old angle should still be correct
            #pass
            angle_new=angle_new
            self.col_RAM[c_id,self.i_old_angle]=- self.col_RAM[c_id,self.i_old_angle] # doesn't help anymore to correct
            if c_id == index:
                print("c,g,k", angle_new*deg, self.col_RAM[c_id,self.i_old_angle]*deg)
        elif d == np.min([a, b, c, d, e,f,g,h,i,j,k,l]) or h == np.min([a, b, c, d, e,f,g,h,i,j,k,l]) or l == np.min([a, b, c, d, e,f,g,h,i,j,k,l]):  # should not be called because the sign of the old angle should still be correct
            angle_new = - angle_new
            self.col_RAM[c_id,self.i_old_angle]= - self.col_RAM[c_id,self.i_old_angle] # doesn't help anymore to correct
            if c_id == index:
                print("d,h,l", angle_new*deg, self.col_RAM[c_id,self.i_old_angle]*deg)
        else:
            raise Exception("vector contains no minimum")

        return angle_new

    def gif_written_info_data(self): #list of strings for every slice step ;later gets reduced to list of strings for every write_interval
        return self.written_info_data

    def calc_action(self, colloids) -> typing.List[Action]:
        actions = []

        for colloid in colloids:
            if colloid.type not in self.acts_on_types:
                actions.append(Action())
                continue

            # setup phases
            if sum(self.col_RAM[colloid.id, :self.n_phases]) == 0:
                raise Exception("Colloid is in no phase.")
            else:
                # print(self.col_RAM[colloid.id, :self.n_phases])
                [phase], = np.where(self.col_RAM[colloid.id, :self.n_phases] != 0)
            index =self.index

            if not self.experiment_engine and colloid.id == index:
                written_info_string = f"phase =" + str(phase)+ f"\n len = " + str(self.col_RAM[index, phase]) + f"\n old_ang=" + str(round(180/np.pi*self.col_RAM[index, self.i_old_angle])) + f"\n steeran = " + str(round(180/np.pi*self.col_RAM[index, self.i_angle_steer]))
                self.written_info_data.append(written_info_string)

            # self.col_RAM[colloid.id,:self.n_phases] = self.col_RAM[colloid.id,:self.n_phases] * 0
            # self.col_RAM[colloid.id,phase] = self.phase_len[colloid.id,0]-1

            # record schmell_magnitude every time in whatever phase
            schmell_magnitude = 0
            #for rod_colloid in colloids:
                #if rod_colloid.id in self.rod_schmell_part_id:
            a, _ = calc_schmell([500,500], colloid.pos[:2])  # die z richtung glitch herum deshalb wird sie hier rausgelassen
            schmell_pos =  [500, 500]
            schmell_magnitude += a

            # gather information when measurement process is running
            if self.col_RAM[colloid.id, self.i_measure_count] != 0:
                self.gather_measurement(colloid.id, schmell_magnitude)
                #print("z999 gather information when measurement process is running","phase",phase,"phase_len",self.phase_len[colloid.id,:],"col_RAM",self.col_RAM[colloid.id,:])

            # remember schmell_magnitude every time in whatever phase
            self.col_RAM[colloid.id, self.i_old_schmell_magnitude] = schmell_magnitude


            # Begin measurement with no history available
            if phase == self.i_observe_think and self.col_RAM[colloid.id, self.i_measure_count] == 0 and self.col_RAM[colloid.id, self.i_start_up] == 0:
                self.col_RAM[colloid.id,self.i_schmell_diff_psum] = 0
                self.col_RAM[colloid.id, self.i_schmell_psum] = schmell_magnitude
                self.gather_measurement(colloid.id, schmell_magnitude)
                self.col_RAM[colloid.id, self.i_start_up] = 1
                #if colloid.id == self.index :
                #    print("z1010  Begin measurement with no history available","phase",phase,"phase_len",self.phase_len[colloid.id,:],"col_RAM",self.col_RAM[colloid.id,:])


                    # End measurement with no history available as soon as phase_len[self.i_observe_think] is full complete measurement and run in random direction
            elif self.col_RAM[colloid.id, self.i_measure_count] == self.phase_len[colloid.id,self.i_observe_think] and self.col_RAM[colloid.id, self.i_start_up] == 1:
                #if colloid.id == self.index:
                #    print("z1015  End measurement with no history available", "phase", phase,"phase_len",self.phase_len[colloid.id,:], "col_RAM",
                #          self.col_RAM[colloid.id, :])

                self.col_RAM[colloid.id, self.i_old_schmell_pot], self.col_RAM[colloid.id, self.i_old_schmell_pot_grad] = self.calc_pot_gradpot(colloid.id, colloid.pos, schmell_pos)
                self.col_RAM[colloid.id,self.i_schmell_diff_psum] = 0
                self.col_RAM[colloid.id, self.i_schmell_psum] = 0

                self.col_RAM[colloid.id, self.i_measure_count] = 0
                self.col_RAM[colloid.id, phase] = 0
                phase = self.i_run
                self.col_RAM[colloid.id, phase] = 1
                self.col_RAM[colloid.id, self.i_start_up] = 2
                #if colloid.id == self.index:
                #    print("z1023  End measurement with no history available and run", "phase", phase,"phase_len",self.phase_len[colloid.id,:], "col_RAM",
                #        self.col_RAM[colloid.id, :])


                    # Begin measurement with one Measurement already taken
            elif phase == self.i_observe_think and self.col_RAM[colloid.id, self.i_measure_count] == 0 and self.col_RAM[colloid.id, self.i_start_up] == 2:
                #if colloid.id == self.index:
                #    print("z1029  Begin measurement with one Measurement already taken", "phase", phase,"phase_len",self.phase_len[colloid.id,:], "col_RAM",
                #        self.col_RAM[colloid.id, :])
                self.col_RAM[colloid.id, self.i_schmell_psum] = schmell_magnitude
                self.gather_measurement(colloid.id, schmell_magnitude)
                self.col_RAM[colloid.id, phase] = 0
                phase = self.i_steer_right
                self.col_RAM[colloid.id, phase] = 1
                # Doing wild quess which direction to steer
                if self.col_RAM[colloid.id, self.i_old_schmell_magnitude] > self.col_RAM[colloid.id, self.i_old_schmell_pot]:  # roughly nice direction
                    self.phase_len[colloid.id,self.i_steer_right] = int(
                        np.ceil(90 * np.pi / 180 / self.steer_speed))  # self.steer_speed rad/step
                    self.col_RAM[colloid.id, self.i_angle_steer] = -90 * np.pi / 180
                else:  # roughly ugly direction
                    self.phase_len[colloid.id,self.i_steer_right] = int(np.ceil(120 * np.pi / 180 / self.steer_speed))
                    self.col_RAM[colloid.id, self.i_angle_steer] = -120 * np.pi / 180
                self.col_RAM[colloid.id, self.i_start_up] = 3
                #if colloid.id == self.index:
                #    print("z1043  Begin measurement with one Measurement already quess steering", "phase", phase,"phase_len",self.phase_len[colloid.id,:], "col_RAM",
                #        self.col_RAM[colloid.id, :])

            elif phase == self.i_observe_think and self.col_RAM[colloid.id, self.i_measure_count] >= self.phase_len[colloid.id,self.i_observe_think] and self.col_RAM[colloid.id, self.i_start_up] == 3:
                #if colloid.id == self.index:
                #    print("z1055 End second measurement   ", "phase", phase,
                #        "phase_len", self.phase_len[colloid.id,:], "col_RAM", self.col_RAM[colloid.id, :])
                schmell_pot, schmell_pot_grad = self.calc_pot_gradpot(colloid.id, colloid.pos, schmell_pos)
                self.col_RAM[colloid.id,self.i_schmell_diff_psum] = 0
                self.col_RAM[colloid.id, self.i_schmell_psum] = 0

                val = 2 * (schmell_pot - self.col_RAM[colloid.id, self.i_old_schmell_pot]) / (self.len_run * (
                            schmell_pot_grad + self.col_RAM[colloid.id, self.i_old_schmell_pot_grad]))
                angle_new = np.arccos(min(max(val, -1), 1))  # restrict to convertable numbers
                if angle_new!=0:
                    self.col_RAM[colloid.id, self.i_old_angle] = angle_new
                else:
                    self.col_RAM[colloid.id, self.i_old_angle] = angle_new+1e-6
                self.col_RAM[colloid.id, self.i_old_schmell_pot] = schmell_pot
                self.col_RAM[colloid.id, self.i_old_schmell_pot_grad] = schmell_pot_grad

                self.col_RAM[colloid.id, self.i_measure_count] = 0
                self.col_RAM[colloid.id, phase] = 0
                phase = self.i_run
                self.col_RAM[colloid.id, phase] = 1
                self.col_RAM[colloid.id, self.i_start_up] = 4
                #if colloid.id == self.index:
                #    print("z1070 End second measurement and start run  ", "phase", phase,
                #        "phase_len", self.phase_len[colloid.id,:], "col_RAM", self.col_RAM[colloid.id, :])




            ################################################################################################################################
            # If the zick zack iterates after the setup of the first zick zack is done i.e. self.col_RAM[colloid.id,self.i_old_angle]!=0
            if self.col_RAM[colloid.id, self.i_start_up] == 4:
                # Begin measurement and determine which steering phase fits, seems like you just stopped running.
                if phase == self.i_observe_think and self.col_RAM[colloid.id, self.i_measure_count] == 0:
                    #if colloid.id == self.index:
                    #    print("z1081 iterate arrive and quess steering start measuring  ", "phase", phase,
                    #        "phase_len", self.phase_len[colloid.id,:], "col_RAM", self.col_RAM[colloid.id, :])
                    self.col_RAM[colloid.id, self.i_schmell_psum] = schmell_magnitude
                    self.gather_measurement(colloid.id, schmell_magnitude)
                    if self.col_RAM[colloid.id, self.i_old_angle] - self.col_RAM[colloid.id, self.i_angle_steer] <= 0:
                        self.col_RAM[colloid.id, phase] = 0
                        phase = self.i_steer_right
                        self.col_RAM[colloid.id, phase] = 1
                        self.phase_len[colloid.id,self.i_steer_right] = int(np.ceil(
                            abs(self.col_RAM[colloid.id, self.i_old_angle] - self.col_RAM[colloid.id, self.i_angle_steer]) / (
                                self.steer_speed)))  # self.steer_speed rad/step
                    elif self.col_RAM[colloid.id, self.i_old_angle]-self.col_RAM[colloid.id, self.i_angle_steer] > 0:
                        self.col_RAM[colloid.id, phase] = 0
                        phase = self.i_steer_left
                        self.col_RAM[colloid.id, phase] = 1
                        self.phase_len[colloid.id,self.i_steer_left] = int(np.ceil(
                            abs(self.col_RAM[colloid.id, self.i_old_angle] - self.col_RAM[colloid.id, self.i_angle_steer]) / (
                                self.steer_speed)))  # self.steer_speed rad/step
                    #if colloid.id == self.index:
                    #    print("z1098 iterate arrive and quess steering start measuring  ", "phase", phase,
                    #        "phase_len", self.phase_len[colloid.id,:], "col_RAM", self.col_RAM[colloid.id, :])

                # If the measurements are gathered it's time for analysis.
                if self.col_RAM[colloid.id, self.i_measure_count] == self.phase_len[colloid.id,self.i_observe_think]:
                    #if colloid.id == self.index:
                    #    print("z1103 iterate measurement done analyse", "phase", phase,
                    #        "phase_len", self.phase_len[colloid.id,:], "col_RAM", self.col_RAM[colloid.id, :])

                    schmell_pot, schmell_pot_grad = self.calc_pot_gradpot(colloid.id, colloid.pos, schmell_pos)

                    val = 2 * (schmell_pot - self.col_RAM[colloid.id, self.i_old_schmell_pot]) / (
                                self.len_run * (schmell_pot_grad + self.col_RAM[colloid.id, self.i_old_schmell_pot_grad]))
                    angle_new = np.arccos(min(max(val, -1), 1))  # restrict to convertable numbers

                    angle_new = self.resolve_sign(colloid.id, angle_new)

                    # gather already moved angle in steps
                    if phase == self.i_observe_think:
                        if self.col_RAM[colloid.id, self.i_old_angle] - self.col_RAM[colloid.id, self.i_angle_steer] >= 0:
                            angle_steps_done = self.phase_len[colloid.id,self.i_steer_left]
                        elif self.col_RAM[colloid.id, self.i_old_angle]-self.col_RAM[colloid.id, self.i_angle_steer] < 0:
                            angle_steps_done = -self.phase_len[colloid.id,self.i_steer_right]
                    elif phase == self.i_steer_right:
                        angle_steps_done = -self.col_RAM[colloid.id, self.i_steer_right]
                    elif phase == self.i_steer_left:
                        angle_steps_done = self.col_RAM[colloid.id, self.i_steer_left]

                    # determine final angle


                    if angle_new < 0 :
                        angle_steps_to_do = int(np.ceil(
                            (angle_new - self.zick_angle) / (self.steer_speed)))  # self.steer_speed rad/step
                    elif angle_new > 0 :
                        angle_steps_to_do = int(np.ceil(
                            (angle_new + self.zick_angle) / (self.steer_speed)))  # self.steer_speed rad/step
                    else:
                        angle_new+=1e-5 #prevent that angle_new=0 because this shall only be valid at the beginning of a zickzack path #depreciated
                        angle_steps_to_do = int(np.ceil(
                            (angle_new + self.zick_angle) / (self.steer_speed)))


                    angle_steps_left_of = angle_steps_to_do - angle_steps_done
                    if angle_steps_left_of < 0:
                        self.col_RAM[colloid.id, phase] = 0
                        phase = self.i_steer_right
                        self.col_RAM[colloid.id, phase] = 1
                        self.phase_len[colloid.id,self.i_steer_right] = int(abs(angle_steps_left_of))
                    elif angle_steps_left_of > 0:
                        self.col_RAM[colloid.id, phase] = 0
                        phase = self.i_steer_left
                        self.col_RAM[colloid.id, phase] = 1
                        self.phase_len[colloid.id,self.i_steer_left] = int(abs(angle_steps_left_of))
                    else:
                        self.col_RAM[colloid.id, phase] = 0
                        phase = self.i_observe_think
                        self.col_RAM[colloid.id, phase] = 1


                    self.col_RAM[colloid.id, self.i_old_angle] = angle_new
                    self.col_RAM[colloid.id, self.i_angle_steer] = angle_steps_to_do * self.steer_speed  # self.steer_speed rad/step
                    #if colloid.id == self.index:
                    #    print("z1143 iterate analyse done  angle fixed final turning initialized", "phase", phase,
                    #        "phase_len", self.phase_len[colloid.id,:], "col_RAM", self.col_RAM[colloid.id, :])

                # finally steering has finished, close measurement process, memorizes results and get back running
                if self.col_RAM[colloid.id, self.i_measure_count] >= self.phase_len[colloid.id,self.i_observe_think] and phase == self.i_observe_think:
                    #if colloid.id == self.index:
                    #    print("z1148 iterate final turning done measurement refining", "phase", phase,
                    #        "phase_len", self.phase_len[colloid.id,:], "col_RAM", self.col_RAM[colloid.id, :])
                    self.col_RAM[colloid.id, self.i_old_schmell_pot], self.col_RAM[
                        colloid.id, self.i_old_schmell_pot_grad] = self.calc_pot_gradpot(colloid.id,colloid.pos, schmell_pos)
                    self.col_RAM[colloid.id, self.i_schmell_diff_psum] = 0
                    self.col_RAM[colloid.id, self.i_schmell_psum] = 0

                    self.col_RAM[colloid.id, self.i_measure_count] = 0
                    self.col_RAM[colloid.id, phase] = 0
                    phase = self.i_run
                    self.col_RAM[colloid.id, phase] = 1
                    #if colloid.id == self.index:
                    #    print("z1157 iterate final turning done measurement refining, run again", "phase", phase,
                    #        "phase_len", self.phase_len[colloid.id,:], "col_RAM", self.col_RAM[colloid.id, :])

            #################################################################################
            # actually doing stuff
            if phase == self.i_run:
                force_mult = 1
                torque_z = 0
                #print("ich war hier", self.col_RAM[colloid.id,self.i_run])
            elif phase == self.i_steer_left:
                force_mult = 0
                torque_z = 1
            elif phase == self.i_steer_right:
                force_mult = 0
                torque_z = -1
            elif phase == self.i_observe_think:
                force_mult = 0
                torque_z = 0
            else:
                raise Exception("Colloid doesn't know what to do. Unexpected phase identifier selected")

            actions.append(
                Action(force=self.act_force * force_mult, torque=np.array([0, 0, self.act_torque * torque_z]))
            )

            #################################################################################
            # propagate phases
            for j in range(self.n_phases):
                if phase == j and self.col_RAM[colloid.id, phase] == self.phase_len[colloid.id,j]:
                    self.col_RAM[colloid.id, phase] = 0
                    phase = self.phase_trans[j]

            self.col_RAM[colloid.id, phase] += 1

        # print(self.col_RAM[1])

        return actions

class rotate_rod_vision_cone(InteractionModel):
    def __init__(
            self,
            data_folder = "/work/skoppenhoefer",
            act_force=42,
            act_torque=42,
            n_type=[],
            rod_particle_type=42,
            radius_vector = [42,42,42],
            detection_radius_position=1.0,
            vision_half_angle=np.pi / 2.0,
            n_cones=5,
            phase_len=[42,42,42,42],
            experiment_engine=False,
            acts_on_types: typing.List[int] = None,
    ):
        self.act_force = act_force
        self.act_torque = act_torque
        self.n_type = n_type
        self.detection_radius_position = detection_radius_position
        self.vision_half_angle = vision_half_angle
        self.n_cones = n_cones
        self.rod_particle_type = rod_particle_type
        self.radius_vector = radius_vector # this it the perceived radius for the vision cones
        
        self.phase_len = phase_len
        self.vision_cone_data = []
        self.written_info_data = None
        self.experiment_engine = experiment_engine
        if acts_on_types is None:
            acts_on_types = [0]
        self.acts_on_types = acts_on_types
        if self.experiment_engine==False:
            files = os.listdir(data_folder)
            if "written_info_data.txt" in files:
                os.remove(data_folder + "/written_info_data.txt")
            self.written_info_data_file = open(data_folder + "/written_info_data.txt",'a')
            if "vision_cone_data.pick" in files:
                os.remove(data_folder + "/vision_cone_data.pick")
            self.vision_cone_data_file = open(data_folder + "/vision_cone_data.pick",'ab')

        self.n_col = sum(self.n_type)
        self.n_phases = 4  # run, steer
        self.n_memory = 2  # schmellmemory
        self.col_RAM = np.zeros((self.n_col, self.n_phases + self.n_memory))
        self.col_RAM[:,0] = 1 # starting in phase 0 with a single step

        self.vision_handle= SubdividedVisionCones(self.detection_radius_position, self.vision_half_angle, self.n_cones, self.radius_vector)





        self.phase_trans = [0,0,0,3]

        #print(self.col_RAM[1])

    def close_written_info_data_file(self):
        self.written_info_data_file.close()

    def close_vision_cone_data_file(self):
        self.vision_cone_data_file.close()


    def calc_action(self, colloids) -> typing.List[Action]:
        actions = []
        # fig, ax = plt.subplots(1,1)
        cone_data = []
        for colloid in colloids:
            if colloid.type not in self.acts_on_types:
                actions.append(Action())
                continue

            #setup phases
            if sum(self.col_RAM[colloid.id, :self.n_phases]) == 0:
                phase = 0
            else:
                #print(self.col_RAM[colloid.id, :self.n_phases])
                [phase], = np.where(self.col_RAM[colloid.id, :self.n_phases] != 0)



            vision_vals = self.vision_handle.compute_observable(colloid,colloids)
            cone_data.append([colloid.id, vision_vals])


            if self.experiment_engine==False and colloid.id == 0:
                written_info_string = 'green '+str(np.round(np.array(vision_vals[:,1])*100,0))+'*e-2'
                self.written_info_data_file.write(written_info_string + '\n')

            vision_vals_boolean= list(np.where(vision_vals[:,1]>0, True, False))


            if vision_vals_boolean[0]==True :
                self.col_RAM[colloid.id, phase]=0
                phase=2  #right
                self.col_RAM[colloid.id, phase] = 1
            elif vision_vals_boolean == [False, True, True, True, True]:
                self.col_RAM[colloid.id, phase] = 0
                phase = 0  #forward
                self.col_RAM[colloid.id, phase] = 1
            elif vision_vals_boolean == [True, True, True, True, True]:
                self.col_RAM[colloid.id, phase]=0
                phase = -1
                self.col_RAM[colloid.id, phase] = 1
            elif vision_vals_boolean == [False, False, True, True, True] or vision_vals_boolean == [False, False, False, True, True]:
                self.col_RAM[colloid.id, phase] = 0
                phase = 1
                self.col_RAM[colloid.id, phase] = 1
            elif vision_vals_boolean == [False, False, False, False, True]:
                self.col_RAM[colloid.id, phase] = 0
                phase = 1  # left
                self.col_RAM[colloid.id, phase] = 1
            elif vision_vals_boolean == [False, False, False, False, False]:
                self.col_RAM[colloid.id, phase]=0
                phase = 1  #left
                self.col_RAM[colloid.id, phase] = 1
            else:
                self.col_RAM[colloid.id, phase] = 0
                phase = 0  #forward
                self.col_RAM[colloid.id, phase] = 1


            if phase==0: #run
                force_mult = 1
                torque_z =  0
            elif phase==1: #steer left
                force_mult = 0.25/0.6 * int(not self.experiment_engine) # even when turning there is a forward motion
                torque_z =  1
            elif phase==2: # steer ang right
                force_mult = 0.25/0.6 * int(not self.experiment_engine)
                torque_z = -1
            elif phase==3: # nothing
                force_mult = 0
                torque_z = 0
            else:
                raise Exception("Colloid doesn't know what to do. Unexpected phase identifier selected")

            actions.append(
                Action(force=self.act_force * force_mult, torque=np.array([0, 0, self.act_torque * torque_z]))
            )

            #propagate phases
            for j in range(self.n_phases):
                if phase == j and self.col_RAM[colloid.id, phase] == self.phase_len[j]:
                    self.col_RAM[colloid.id, phase] = 0
                    phase = self.phase_trans[j]

            self.col_RAM[colloid.id, phase] += 1

        #print(self.col_RAM[1])
        if self.experiment_engine==False:
            pickle.dump(cone_data, self.vision_cone_data_file)

        return actions


class rotate_rod_vision_cone_interaction(InteractionModel):
    def __init__(
            self,
            data_folder = "/work/skoppenhoefer",
            act_force=42,
            act_torque=42,
            n_type=[],
            rod_particle_type=42,
            radius_vector = [42,42,42],
            detection_radius_position=1.0,
            vision_half_angle=np.pi / 2.0,
            n_cones=5,
            phase_len=[42,42,42,42],
            experiment_engine=False,
            acts_on_types: typing.List[int] = None,
    ):
        self.act_force = act_force
        self.act_torque = act_torque
        self.n_type = n_type
        self.detection_radius_position = detection_radius_position
        self.vision_half_angle = vision_half_angle
        self.n_cones = n_cones
        self.rod_particle_type = rod_particle_type
        self.radius_vector = radius_vector # this it the perceived radius for the vision cones
        self.phase_len = phase_len
        self.vision_cone_data = []
        self.written_info_data = None
        self.experiment_engine = experiment_engine
        if acts_on_types is None:
            acts_on_types = [0]
        self.acts_on_types = acts_on_types

        if self.experiment_engine==False:
            files = os.listdir(data_folder)
            if "written_info_data.txt" in files:
                os.remove(data_folder + "/written_info_data.txt")
            self.written_info_data_file = open(data_folder + "/written_info_data.txt",'a')
            if "vision_cone_data.pick" in files:
                os.remove(data_folder + "/vision_cone_data.pick")
            self.vision_cone_data_file = open(data_folder + "/vision_cone_data.pick",'ab')

        self.n_col = sum(self.n_type)
        self.n_phases = 4  # run, steer
        self.n_memory = 2  # schmellmemory
        self.col_RAM = np.zeros((self.n_col, self.n_phases + self.n_memory))
        self.col_RAM[:,0] = 1 # starting in phase 0 with a single step


        self.vision_handle= SubdividedVisionCones(self.detection_radius_position,self.vision_half_angle,self.n_cones,self.radius_vector)



        self.phase_trans = [0,0,0,3]

        #print(self.col_RAM[1])

    def close_written_info_data_file(self):
        self.written_info_data_file.close()

    def close_vision_cone_data_file(self):
        self.vision_cone_data_file.close()


    def calc_action(self, colloids) -> typing.List[Action]:
        actions = []
        # fig, ax = plt.subplots(1,1)
        cone_data = []
        for colloid in colloids:
            if colloid.type not in self.acts_on_types:
                actions.append(Action())
                continue

            #setup phases
            if sum(self.col_RAM[colloid.id, :self.n_phases]) == 0:
                phase = 0
            else:
                #print(self.col_RAM[colloid.id, :self.n_phases])
                [phase], = np.where(self.col_RAM[colloid.id, :self.n_phases] != 0)



            vision_vals = self.vision_handle.compute_observable(colloid,colloids)
            cone_data.append([colloid.id, vision_vals])


            if self.experiment_engine==False and colloid.id == self.n_type[0]-1:
                written_info_string = 'green '+str(np.round(np.array(vision_vals[:,1])*100,0))+ ' *e-2 | red '+ str(np.round(np.array(vision_vals[:,0])*100,0))+' *e-2'
                self.written_info_data_file.write(written_info_string + '\n')

            vision_vals_boolean= list(np.where(vision_vals[:,1]>0, True, False))

            vision_vals_interaction_boolean = list(np.where(vision_vals[:, 0] > 0, True, False))

            if vision_vals_boolean[1:4] == [True, True, True] and vision_vals[0,0] > 0.02:
                #if your pushing the rod and see a friend on the right steer away, to not block each other
                self.col_RAM[colloid.id, phase] = 0
                phase = 1  #left
                self.col_RAM[colloid.id, phase] = 1
            elif vision_vals_boolean[0]==True :
                self.col_RAM[colloid.id, phase]=0
                phase = 2  #right
                self.col_RAM[colloid.id, phase] = 1
            elif vision_vals_boolean == [False, True, True, True, True]:
                self.col_RAM[colloid.id, phase] = 0
                phase = 0  #forward
                self.col_RAM[colloid.id, phase] = 1
            elif vision_vals_boolean == [True, True, True, True, True]:
                self.col_RAM[colloid.id, phase]=0
                phase = -1
                self.col_RAM[colloid.id, phase] = 1
            elif vision_vals_boolean == [False, False, True, True, True] or vision_vals_boolean == [False, False, False, True, True]:
                self.col_RAM[colloid.id, phase] = 0
                phase = 1
                self.col_RAM[colloid.id, phase] = 1
            elif vision_vals_boolean == [False, False, False, False, True]:
                self.col_RAM[colloid.id, phase] = 0
                phase = 1  # left
                self.col_RAM[colloid.id, phase] = 1
            elif vision_vals_boolean == [False, False, False, False, False]:
                self.col_RAM[colloid.id, phase]=0
                phase = 1  #left
                self.col_RAM[colloid.id, phase] = 1
            else:
                self.col_RAM[colloid.id, phase] = 0
                phase = 0  #forward
                self.col_RAM[colloid.id, phase] = 1


            if phase==0: #run
                force_mult = 1
                torque_z =  0
            elif phase==1: #steer left
                force_mult = 0.25/0.6 * int( not self.experiment_engine) # even when turning there is a forward motion
                torque_z =  1
            elif phase==2: # steer ang right
                force_mult = 0.25/0.6 * int( not self.experiment_engine)
                torque_z = -1
            elif phase==3: # nothing
                force_mult = 0
                torque_z = 0
            else:
                raise Exception("Colloid doesn't know what to do. Unexpected phase identifier selected")

            actions.append(
                Action(force=self.act_force * force_mult, torque=np.array([0, 0, self.act_torque * torque_z]))
            )

            #propagate phases
            for j in range(self.n_phases):
                if phase == j and self.col_RAM[colloid.id, phase] == self.phase_len[j]:
                    self.col_RAM[colloid.id, phase] = 0
                    phase = self.phase_trans[j]

            self.col_RAM[colloid.id, phase] += 1

        #print(self.col_RAM[1])
        if self.experiment_engine==False:
            pickle.dump(cone_data, self.vision_cone_data_file)

        return actions


def get_vision_of_colloid( colloid, colloids, radii, vision_half_angle, n_cones, vision_range ): # in 2D
    # in 2D, radii contains the radius by colloid.id
    types = []
    #determine how many spezies there are, to see them all independently
    for c in colloids: 
        if c.type not in types:
            types.append(c.type)

    my_director = colloid.director[:2]
    vision_val_out = np.zeros((n_cones, len(types)))
    for c in colloids:
        dist = c.pos[:2]-colloid.pos[:2]
        dist_norm = np.linalg.norm(dist)
        # don't see yourself, set vision_range sufficiently high if no upper limit is wished
        if dist_norm != 0 and dist_norm < vision_range:
            # calc perceived angle deviation (sign of angle is missing)
            angle = np.arccos(np.dot(dist / dist_norm, my_director))
            # use the director in orthogonal direction to determine sign
            orthogonal_dot=np.dot(dist / dist_norm, [-my_director[1], my_director[0]])
            if orthogonal_dot==0:###########
                orthogonal_dot=1########### maybe replace with angle*=np.sign(orthogonal_dot)
            angle *= np.sign(orthogonal_dot)
            # sort the perceived colloid c by their vision cone and type 

            for cone in range(n_cones):
                if (-vision_half_angle + cone * vision_half_angle *2 /n_cones < angle and angle < -vision_half_angle + (cone+1) * vision_half_angle *2/n_cones):
                    type_num = np.where(np.array(types) == c.type)[0][0]
                    vision_val_out[cone,type_num] += np.min([1, 2*radii[c.id]/dist_norm])
    return vision_val_out


class zickzack_pointfind_maze(InteractionModel):
    def __init__(
            self,
            act_force=42,
            act_torque=42,
            n_type=[],
            schmell_pos=[42,42],
            phase_len=[30, 420, 420, 64],
            diffusion_coeff=42,
            zick_angle=20,
            acts_on_types: typing.List[int] = None,
    ):
        self.act_force = act_force
        self.act_torque = act_torque
        self.n_type = n_type
        self.schmell_pos = schmell_pos

        self.steer_speed = 0.8 * np.pi / 180  # rad per step
        self.len_run = 20  # mu

        self.diffusion_coeff = 1.4 # mu ^2 per second transversal
        self.t_step = 0.2 #second
        self.zick_angle = zick_angle * np.pi / 180
        self.written_info_data = []


        # steer_angle=36*np.pi/180   # 36 degree
        # steer_steps=int(np.ceil(steer_angle/self.steer_speed)) # self.steer_speed rad/step
        # run_distance=4 # mu
        # run_steps = int(np.ceil(run_distance/0.1)) # mu per step
        # self.phase_len = [run_steps,steer_steps] #run ,steer

        if acts_on_types is None:
            acts_on_types = [0]
        self.acts_on_types = acts_on_types

        self.n_col = sum(self.n_type)
        self.n_phases = 4  # run, steer_left, steer_right, observe_think,
        self.n_memory = 9  # measure_count, schmellpartialsum, last schnellmagnitude, schelldifferenzpartialsum, last schmell potential, last schmell potential gradient,old_angle,angle_steer, start up counter
        self.col_RAM = np.zeros((self.n_col, self.n_phases + self.n_memory))
        self.phase_len = np.ones((self.n_col, len(phase_len)))
        self.phase_len[:,0] *= int(self.len_run / 0.1 ) # 0.1 mu/step
        self.phase_len[:,1] *= phase_len[1]
        self.phase_len[:, 2] *= phase_len[2]
        self.phase_len[:, 3] *= phase_len[3]
        self.col_RAM[:, 3] = 1  # starting in phase 0 with a single step

        self.i_run = 0
        self.i_steer_left = 1
        self.i_steer_right = 2
        self.i_observe_think = 3
        self.i_measure_count = 4
        self.i_schmell_psum = 5
        self.i_old_schmell_magnitude = 6
        self.i_schmell_diff_psum = 7
        self.i_old_schmell_pot = 8
        self.i_old_schmell_pot_grad = 9
        self.i_old_angle = 10
        self.i_angle_steer = 11
        self.i_start_up = 12
        self.index = 0

        self.phase_trans = [3, 3, 3, 0]

        # print(self.col_RAM[1])

    def gather_measurement(self, c_id, schmell_magnitude):
        self.col_RAM[c_id, self.i_schmell_diff_psum] += abs(schmell_magnitude - self.col_RAM[c_id, self.i_old_schmell_magnitude])
        self.col_RAM[c_id, self.i_schmell_psum] += schmell_magnitude
        self.col_RAM[c_id, self.i_measure_count] += 1

    def calc_pot_gradpot(self, c_id , c_pos, c_rod_pos):
        schmell_pot = self.col_RAM[c_id, self.i_schmell_psum] / (self.col_RAM[c_id, self.i_measure_count]+1) # There is one more measurement done here
        mystical_factor=5.5
        schmell_pot_grad = self.col_RAM[c_id, self.i_schmell_diff_psum]*mystical_factor / (self.col_RAM[c_id, self.i_measure_count]*np.sqrt(4 * self.diffusion_coeff * self.t_step / np.pi ))
        schmell_magnitude = 0
        schmell_grad = 0
        a, b = calc_schmell(c_rod_pos[:2], c_pos[:2])  # die z richtung glitch herum deshalb wird sie hier rausgelassen
        schmell_magnitude = a
        schmell_grad = np.linalg.norm(b)
        if c_id == self.index :
            print(c_pos)
            print("pot", schmell_pot, "ana pot", schmell_magnitude ,"frac",schmell_pot/schmell_magnitude)
            print("grad", schmell_pot_grad, "ana_pot_grad", schmell_grad, "frac",schmell_pot_grad/schmell_grad)

        return schmell_pot, schmell_pot_grad

    def resolve_sign(self, c_id, angle_new):
        index=self.index
        a = abs((1) * abs(self.col_RAM[c_id, self.i_old_angle]) - self.col_RAM[c_id, self.i_angle_steer] - (1) * angle_new)
        b = abs((1) * abs(self.col_RAM[c_id, self.i_old_angle]) - self.col_RAM[c_id, self.i_angle_steer] - (-1) * angle_new)
        c = abs((-1) * abs(self.col_RAM[c_id, self.i_old_angle]) - self.col_RAM[c_id, self.i_angle_steer] - (1) * angle_new)
        d = abs((-1) * abs(self.col_RAM[c_id, self.i_old_angle]) - self.col_RAM[c_id, self.i_angle_steer] - (-1) * angle_new)

        e = abs((1) * abs(self.col_RAM[c_id, self.i_old_angle]) - self.col_RAM[c_id, self.i_angle_steer] - (1) * angle_new+360)
        f = abs((1) * abs(self.col_RAM[c_id, self.i_old_angle]) - self.col_RAM[c_id, self.i_angle_steer] - (-1) * angle_new+360)
        g = abs((-1) * abs(self.col_RAM[c_id, self.i_old_angle]) - self.col_RAM[c_id, self.i_angle_steer] - (1) * angle_new+360)
        h = abs((-1) * abs(self.col_RAM[c_id, self.i_old_angle]) - self.col_RAM[c_id, self.i_angle_steer] - (-1) * angle_new+360)

        i = abs((1) * abs(self.col_RAM[c_id, self.i_old_angle]) - self.col_RAM[c_id, self.i_angle_steer] - (1) * angle_new-360)
        j = abs((1) * abs(self.col_RAM[c_id, self.i_old_angle]) - self.col_RAM[c_id, self.i_angle_steer] - (-1) * angle_new-360)
        k = abs((-1) * abs(self.col_RAM[c_id, self.i_old_angle]) - self.col_RAM[c_id, self.i_angle_steer] - (1) * angle_new-360)
        l = abs((-1) * abs(self.col_RAM[c_id, self.i_old_angle]) - self.col_RAM[c_id, self.i_angle_steer] - (-1) * angle_new-360)

        deg=180/np.pi
        if c_id == index:
            print(str(round(a*deg)) + "= a=|(1)  *" + str(round(abs(self.col_RAM[c_id, self.i_old_angle])*deg)) + "-(" + str(round(self.col_RAM[c_id, self.i_angle_steer]*deg)) + ")- (1 ) *" + str(round(angle_new*deg)) + "|")
            print(str(round(b*deg)) + "= b=|(1)  *" + str(round(abs(self.col_RAM[c_id, self.i_old_angle])*deg)) + "-(" + str(round(self.col_RAM[c_id, self.i_angle_steer]*deg)) + ")- (-1) *" + str(round(angle_new*deg)) + "|")
            print(str(round(c*deg)) + "= c=|(-1) *" + str(round(abs(self.col_RAM[c_id, self.i_old_angle])*deg)) + "-(" + str(round(self.col_RAM[c_id, self.i_angle_steer]*deg)) + ")- (1 ) *" + str(round(angle_new*deg)) + "|")
            print(str(round(d*deg)) + "= d=|(-1) *" + str(round(abs(self.col_RAM[c_id, self.i_old_angle])*deg)) + "-(" + str(round(self.col_RAM[c_id, self.i_angle_steer]*deg)) + ")- (-1) *" + str(round(angle_new*deg)) + "|")

        if a == np.min([a, b, c, d, e,f,g,h,i,j,k,l]) or e == np.min([a, b, c, d, e,f,g,h,i,j,k,l]) or i == np.min([a, b, c, d, e,f,g,h,i,j,k,l]):
            #pass
            angle_new=angle_new
            self.col_RAM[c_id,self.i_old_angle]=self.col_RAM[c_id,self.i_old_angle]
            if c_id == index:
                print("a,e,i", angle_new*deg, self.col_RAM[c_id,self.i_old_angle]*deg)
        elif b == np.min([a, b, c, d, e,f,g,h,i,j,k,l]) or f == np.min([a, b, c, d, e,f,g,h,i,j,k,l]) or j == np.min([a, b, c, d, e,f,g,h,i,j,k,l]):
            angle_new = -angle_new
            self.col_RAM[c_id,self.i_old_angle]=self.col_RAM[c_id,self.i_old_angle]
            if c_id == index:
                print("b,f,j", angle_new*deg, self.col_RAM[c_id,self.i_old_angle]*deg)
        elif c == np.min([a, b, c, d, e,f,g,h,i,j,k,l]) or g == np.min([a, b, c, d, e,f,g,h,i,j,k,l]) or k == np.min([a, b, c, d, e,f,g,h,i,j,k,l]):  # should not be called because the sign of the old angle should still be correct
            #pass
            angle_new=angle_new
            self.col_RAM[c_id,self.i_old_angle]=- self.col_RAM[c_id,self.i_old_angle] # doesn't help anymore to correct
            if c_id == index:
                print("c,g,k", angle_new*deg, self.col_RAM[c_id,self.i_old_angle]*deg)
        elif d == np.min([a, b, c, d, e,f,g,h,i,j,k,l]) or h == np.min([a, b, c, d, e,f,g,h,i,j,k,l]) or l == np.min([a, b, c, d, e,f,g,h,i,j,k,l]):  # should not be called because the sign of the old angle should still be correct
            angle_new = - angle_new
            self.col_RAM[c_id,self.i_old_angle]= - self.col_RAM[c_id,self.i_old_angle] # doesn't help anymore to correct
            if c_id == index:
                print("d,h,l", angle_new*deg, self.col_RAM[c_id,self.i_old_angle]*deg)
        else:
            raise Exception("vector contains no minimum")

        return angle_new

    def close_written_info_data_file(self):
        self.written_info_data_file.close()

    def calc_action(self, colloids) -> typing.List[Action]:
        actions = []

        for colloid in colloids:
            if colloid.type not in self.acts_on_types:
                actions.append(Action())
                continue

            # setup phases
            if sum(self.col_RAM[colloid.id, :self.n_phases]) == 0:
                raise Exception("Colloid is in no phase.")
            else:
                # print(self.col_RAM[colloid.id, :self.n_phases])
                [phase], = np.where(self.col_RAM[colloid.id, :self.n_phases] != 0)
            index =self.index
            if colloid.id == index:
                written_info_string = f" phase =" + str(phase)+ f", len = " + str(self.col_RAM[index, phase]) + f", old_ang=" + str(round(180/np.pi*self.col_RAM[index, self.i_old_angle])) + f", steeran = " + str(round(180/np.pi*self.col_RAM[index, self.i_angle_steer]))
                self.written_info_data.write(written_info_string + '\n')
            # self.col_RAM[colloid.id,:self.n_phases] = self.col_RAM[colloid.id,:self.n_phases] * 0
            # self.col_RAM[colloid.id,phase] = self.phase_len[colloid.id,0]-1

            # record schmell_magnitude every time in whatever phase
            schmell_magnitude = 0
            a, _ = calc_schmell(self.schmell_pos, colloid.pos[:2])  # die z richtung glitch herum deshalb wird sie hier rausgelassen
            rod_pos =  self.schmell_pos
            schmell_magnitude += a

            # gather information when measurement process is running
            if self.col_RAM[colloid.id, self.i_measure_count] != 0:
                self.gather_measurement(colloid.id, schmell_magnitude)
                #print("z999 gather information when measurement process is running","phase",phase,"phase_len",self.phase_len[colloid.id,:],"col_RAM",self.col_RAM[colloid.id,:])

            # remember schmell_magnitude every time in whatever phase
            self.col_RAM[colloid.id, self.i_old_schmell_magnitude] = schmell_magnitude


            # Begin measurement with no history available
            if phase == self.i_observe_think and self.col_RAM[colloid.id, self.i_measure_count] == 0 and self.col_RAM[colloid.id, self.i_start_up] == 0:
                self.col_RAM[colloid.id,self.i_schmell_diff_psum] = 0
                self.col_RAM[colloid.id, self.i_schmell_psum] = schmell_magnitude
                self.gather_measurement(colloid.id, schmell_magnitude)
                self.col_RAM[colloid.id, self.i_start_up] = 1
                #if colloid.id == self.index :
                    #print("z1010  Begin measurement with no history available","phase",phase,"phase_len",self.phase_len[colloid.id,:],"col_RAM",self.col_RAM[colloid.id,:])


                    # End measurement with no history available as soon as phase_len[self.i_observe_think] is full complete measurement and run in random direction
            elif self.col_RAM[colloid.id, self.i_measure_count] == self.phase_len[colloid.id,self.i_observe_think] and self.col_RAM[colloid.id, self.i_start_up] == 1:
                #if colloid.id == self.index:
                    #print("z1015  End measurement with no history available", "phase", phase,"phase_len",self.phase_len[colloid.id,:], "col_RAM",
                          #self.col_RAM[colloid.id, :])

                self.col_RAM[colloid.id, self.i_old_schmell_pot], self.col_RAM[colloid.id, self.i_old_schmell_pot_grad] = self.calc_pot_gradpot(colloid.id, colloid.pos, rod_pos)
                self.col_RAM[colloid.id,self.i_schmell_diff_psum] = 0
                self.col_RAM[colloid.id, self.i_schmell_psum] = 0

                self.col_RAM[colloid.id, self.i_measure_count] = 0
                self.col_RAM[colloid.id, phase] = 0
                phase = self.i_run
                self.col_RAM[colloid.id, phase] = 1
                self.col_RAM[colloid.id, self.i_start_up] = 2
                #if colloid.id == self.index:
                    #print("z1023  End measurement with no history available and run", "phase", phase,"phase_len",self.phase_len[colloid.id,:], "col_RAM",
                        #self.col_RAM[colloid.id, :])


                    # Begin measurement with one Measurement already taken
            elif phase == self.i_observe_think and self.col_RAM[colloid.id, self.i_measure_count] == 0 and self.col_RAM[colloid.id, self.i_start_up] == 2:
                #if colloid.id == self.index:
                    #print("z1029  Begin measurement with one Measurement already taken", "phase", phase,"phase_len",self.phase_len[colloid.id,:], "col_RAM",
                        #self.col_RAM[colloid.id, :])
                self.col_RAM[colloid.id, self.i_schmell_psum] = schmell_magnitude
                self.gather_measurement(colloid.id, schmell_magnitude)
                self.col_RAM[colloid.id, phase] = 0
                phase = self.i_steer_right
                self.col_RAM[colloid.id, phase] = 1
                # Doing wild quess which direction to steer
                if self.col_RAM[colloid.id, self.i_old_schmell_magnitude] > self.col_RAM[colloid.id, self.i_old_schmell_pot]:  # roughly nice direction
                    self.phase_len[colloid.id,self.i_steer_right] = int(
                        np.ceil(90 * np.pi / 180 / self.steer_speed))  # self.steer_speed rad/step
                    self.col_RAM[colloid.id, self.i_angle_steer] = -90 * np.pi / 180
                else:  # roughly ugly direction
                    self.phase_len[colloid.id,self.i_steer_right] = int(np.ceil(120 * np.pi / 180 / self.steer_speed))
                    self.col_RAM[colloid.id, self.i_angle_steer] = -120 * np.pi / 180
                self.col_RAM[colloid.id, self.i_start_up] = 3
                #if colloid.id == self.index:
                    #print("z1043  Begin measurement with one Measurement already quess steering", "phase", phase,"phase_len",self.phase_len[colloid.id,:], "col_RAM",
                        #self.col_RAM[colloid.id, :])

            elif phase == self.i_observe_think and self.col_RAM[colloid.id, self.i_measure_count] >= self.phase_len[colloid.id,self.i_observe_think] and self.col_RAM[colloid.id, self.i_start_up] == 3:
                #if colloid.id == self.index:
                    #print("z1055 End second measurement   ", "phase", phase,
                        #"phase_len", self.phase_len[colloid.id,:], "col_RAM", self.col_RAM[colloid.id, :])
                schmell_pot, schmell_pot_grad = self.calc_pot_gradpot(colloid.id, colloid.pos, rod_pos)
                self.col_RAM[colloid.id,self.i_schmell_diff_psum] = 0
                self.col_RAM[colloid.id, self.i_schmell_psum] = 0

                val = 2 * (schmell_pot - self.col_RAM[colloid.id, self.i_old_schmell_pot]) / (self.len_run * (
                            schmell_pot_grad + self.col_RAM[colloid.id, self.i_old_schmell_pot_grad]))
                angle_new = np.arccos(min(max(val, -1), 1))  # restrict to convertable numbers
                if angle_new!=0:
                    self.col_RAM[colloid.id, self.i_old_angle] = angle_new
                else:
                    self.col_RAM[colloid.id, self.i_old_angle] = angle_new+1e-6
                self.col_RAM[colloid.id, self.i_old_schmell_pot] = schmell_pot
                self.col_RAM[colloid.id, self.i_old_schmell_pot_grad] = schmell_pot_grad

                self.col_RAM[colloid.id, self.i_measure_count] = 0
                self.col_RAM[colloid.id, phase] = 0
                phase = self.i_run
                self.col_RAM[colloid.id, phase] = 1
                self.col_RAM[colloid.id, self.i_start_up] = 4
                #if colloid.id == self.index:
                    #print("z1070 End second measurement and start run  ", "phase", phase,
                        #"phase_len", self.phase_len[colloid.id,:], "col_RAM", self.col_RAM[colloid.id, :])




            ################################################################################################################################
            # If the zick zack iterates after the setup of the first zick zack is done i.e. self.col_RAM[colloid.id,self.i_old_angle]!=0
            if self.col_RAM[colloid.id, self.i_start_up] == 4:
                # Begin measurement and determine which steering phase fits, seems like you just stopped running.
                if phase == self.i_observe_think and self.col_RAM[colloid.id, self.i_measure_count] == 0:
                    #if colloid.id == self.index:
                        #print("z1081 iterate arrive and quess steering start measuring  ", "phase", phase,
                        # "phase_len", self.phase_len[colloid.id,:], "col_RAM", self.col_RAM[colloid.id, :])
                    self.col_RAM[colloid.id, self.i_schmell_psum] = schmell_magnitude
                    self.gather_measurement(colloid.id, schmell_magnitude)
                    if self.col_RAM[colloid.id, self.i_old_angle] - self.col_RAM[colloid.id, self.i_angle_steer] <= 0:
                        self.col_RAM[colloid.id, phase] = 0
                        phase = self.i_steer_right
                        self.col_RAM[colloid.id, phase] = 1
                        self.phase_len[colloid.id,self.i_steer_right] = int(np.ceil(
                            abs(self.col_RAM[colloid.id, self.i_old_angle] - self.col_RAM[colloid.id, self.i_angle_steer]) / (
                                self.steer_speed)))  # self.steer_speed rad/step
                    elif self.col_RAM[colloid.id, self.i_old_angle]-self.col_RAM[colloid.id, self.i_angle_steer] > 0:
                        self.col_RAM[colloid.id, phase] = 0
                        phase = self.i_steer_left
                        self.col_RAM[colloid.id, phase] = 1
                        self.phase_len[colloid.id,self.i_steer_left] = int(np.ceil(
                            abs(self.col_RAM[colloid.id, self.i_old_angle] - self.col_RAM[colloid.id, self.i_angle_steer]) / (
                                self.steer_speed)))  # self.steer_speed rad/step
                    #if colloid.id == self.index:
                        #print("z1098 iterate arrive and quess steering start measuring  ", "phase", phase,
                            #"phase_len", self.phase_len[colloid.id,:], "col_RAM", self.col_RAM[colloid.id, :])

                # If the measurements are gathered it's time for analysis.
                if self.col_RAM[colloid.id, self.i_measure_count] == self.phase_len[colloid.id,self.i_observe_think]:
                    #if colloid.id == self.index:
                        #print("z1103 iterate measurement done analyse", "phase", phase,
                            #"phase_len", self.phase_len[colloid.id,:], "col_RAM", self.col_RAM[colloid.id, :])

                    schmell_pot, schmell_pot_grad = self.calc_pot_gradpot(colloid.id, colloid.pos, rod_pos)

                    val = 2 * (schmell_pot - self.col_RAM[colloid.id, self.i_old_schmell_pot]) / (
                                self.len_run * (schmell_pot_grad + self.col_RAM[colloid.id, self.i_old_schmell_pot_grad]))
                    angle_new = np.arccos(min(max(val, -1), 1))  # restrict to convertable numbers

                    angle_new = self.resolve_sign(colloid.id, angle_new)

                    # gather already moved angle in steps
                    if phase == self.i_observe_think:
                        if self.col_RAM[colloid.id, self.i_old_angle] - self.col_RAM[colloid.id, self.i_angle_steer] >= 0:
                            angle_steps_done = self.phase_len[colloid.id,self.i_steer_left]
                        elif self.col_RAM[colloid.id, self.i_old_angle]-self.col_RAM[colloid.id, self.i_angle_steer] < 0:
                            angle_steps_done = -self.phase_len[colloid.id,self.i_steer_right]
                    elif phase == self.i_steer_right:
                        angle_steps_done = -self.col_RAM[colloid.id, self.i_steer_right]
                    elif phase == self.i_steer_left:
                        angle_steps_done = self.col_RAM[colloid.id, self.i_steer_left]

                    # determine final angle


                    if angle_new < 0 :
                        angle_steps_to_do = int(np.ceil(
                            (angle_new - self.zick_angle) / (self.steer_speed)))  # self.steer_speed rad/step
                    elif angle_new > 0 :
                        angle_steps_to_do = int(np.ceil(
                            (angle_new + self.zick_angle) / (self.steer_speed)))  # self.steer_speed rad/step
                    else:
                        angle_new+=1e-5 #prevent that angle_new=0 because this shall only be valid at the beginning of a zickzack path #depreciated
                        angle_steps_to_do = int(np.ceil(
                            (angle_new + self.zick_angle) / (self.steer_speed)))


                    angle_steps_left_of = angle_steps_to_do - angle_steps_done
                    if angle_steps_left_of < 0:
                        self.col_RAM[colloid.id, phase] = 0
                        phase = self.i_steer_right
                        self.col_RAM[colloid.id, phase] = 1
                        self.phase_len[colloid.id,self.i_steer_right] = int(abs(angle_steps_left_of))
                    elif angle_steps_left_of > 0:
                        self.col_RAM[colloid.id, phase] = 0
                        phase = self.i_steer_left
                        self.col_RAM[colloid.id, phase] = 1
                        self.phase_len[colloid.id,self.i_steer_left] = int(abs(angle_steps_left_of))
                    else:
                        self.col_RAM[colloid.id, phase] = 0
                        phase = self.i_observe_think
                        self.col_RAM[colloid.id, phase] = 1


                    self.col_RAM[colloid.id, self.i_old_angle] = angle_new
                    self.col_RAM[colloid.id, self.i_angle_steer] = angle_steps_to_do * self.steer_speed  # self.steer_speed rad/step
                    #if colloid.id == self.index:
                        #print("z1143 iterate analyse done  angle fixed final turning initialized", "phase", phase,
                            #"phase_len", self.phase_len[colloid.id,:], "col_RAM", self.col_RAM[colloid.id, :])

                # finally steering has finished, close measurement process, memorizes results and get back running
                if self.col_RAM[colloid.id, self.i_measure_count] >= self.phase_len[colloid.id,self.i_observe_think] and phase == self.i_observe_think:
                    #if colloid.id == self.index:
                        #print("z1148 iterate final turning done measurement refining", "phase", phase,
                            #"phase_len", self.phase_len[colloid.id,:], "col_RAM", self.col_RAM[colloid.id, :])
                    self.col_RAM[colloid.id, self.i_old_schmell_pot], self.col_RAM[
                        colloid.id, self.i_old_schmell_pot_grad] = self.calc_pot_gradpot(colloid.id,colloid.pos, rod_pos)
                    self.col_RAM[colloid.id, self.i_schmell_diff_psum] = 0
                    self.col_RAM[colloid.id, self.i_schmell_psum] = 0

                    self.col_RAM[colloid.id, self.i_measure_count] = 0
                    self.col_RAM[colloid.id, phase] = 0
                    phase = self.i_run
                    self.col_RAM[colloid.id, phase] = 1
                    #if colloid.id == self.index:
                        #print("z1157 iterate final turning done measurement refining, run again", "phase", phase,
                            #"phase_len", self.phase_len[colloid.id,:], "col_RAM", self.col_RAM[colloid.id, :])

            #################################################################################
            # actually doing stuff
            if phase == self.i_run:
                force_mult = 1
                torque_z = 0
                #print("ich war hier", self.col_RAM[colloid.id,self.i_run])
            elif phase == self.i_steer_left:
                force_mult = 0
                torque_z = 1
            elif phase == self.i_steer_right:
                force_mult = 0
                torque_z = -1
            elif phase == self.i_observe_think:
                force_mult = 0
                torque_z = 0
            else:
                raise Exception("Colloid doesn't know what to do. Unexpected phase identifier selected")

            actions.append(
                Action(force=self.act_force * force_mult, torque=np.array([0, 0, self.act_torque * torque_z]))
            )

            #################################################################################
            # propagate phases
            for j in range(self.n_phases):
                if phase == j and self.col_RAM[colloid.id, phase] == self.phase_len[colloid.id,j]:
                    self.col_RAM[colloid.id, phase] = 0
                    phase = self.phase_trans[j]

            self.col_RAM[colloid.id, phase] += 1

        # print(self.col_RAM[1])

        return actions


