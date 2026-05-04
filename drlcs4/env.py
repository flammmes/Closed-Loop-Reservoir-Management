import gymnasium as gym
from gymnasium import spaces
import numpy as np
import subprocess
import os
from opm.simulators import BlackOilSimulator
from opm.io.parser import Parser
from opm.io.ecl_state import EclipseState
from opm.io.schedule import Schedule
from opm.io.summary import SummaryConfig
from opm.io.deck import DeckKeyword
from opm.io.ecl import ESmry, EclFile
from opm.io.ecl import EGrid
from opm.io.ecl import ERst
import matplotlib.pyplot as plt
import shutil
import random
import json


class ReservoirEnv(gym.Env):
    def __init__(self,env_id):
        super(ReservoirEnv, self).__init__()
        self.env_id = env_id
        self.episode_counter = 0

        self.raw_reward_log = []

        print(f"Initializing environment with ID: {self.env_id}")

        # Define action and observation space
        # Actions are the production rates and gas injection rates
        self.action_low = np.array([10, 10, 10, 10, 10, 10], dtype=np.float32)
        self.action_high = np.array([100, 100, 100, 100, 100, 100], dtype=np.float32)

        # Define normalized action space between 0 and 1
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)

        # Observations are the pressure and water saturation for all cells,
        # and well results (gas injection rates, production rates, water production rates, and BHP)
        self.observation_space = spaces.Dict({
            "res_state": spaces.Box(low=0, high=np.inf, shape=(2,4, 163, 120), dtype=np.float32),
            "well_observations": spaces.Box(low=0, high=np.inf, shape=(9,15), dtype=np.float32),
        })

        self.parser = Parser()

        self.steps = 0
        self.max_steps = 80
        # Define the source and destination paths
        if int(self.env_id) in range(65,75):  # Testing case, use TRUE_DECK
            source_file1 = 'realizations/TRUE_DECK0R.DATA'
            source_file2 = 'realizations/TRUE_DECK0.DATA'
        else:  # Training case, select a random SIM_DECK file
            sim_deck_idx = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            source_file1 = f'realizations/SIM_DECK{sim_deck_idx}0R.DATA'
            source_file2 = f'realizations/SIM_DECK{sim_deck_idx}0.DATA'
        dest_folder = f'{self.env_id}'
        
        # Ensure the destination folder exists
        os.makedirs(dest_folder, exist_ok=True)

        # Define the destination file names
        dest_file1 = os.path.join(dest_folder, 'TRUE_DECK0R.DATA')
        dest_file2 = os.path.join(dest_folder, f'TRUE_DECK{self.env_id}_0.DATA')

        # Copy and rename the files
        shutil.copy(source_file1, dest_file1)
        shutil.copy(source_file2, dest_file2)
        self.deck = self.parser.parse(f'{self.env_id}/TRUE_DECK0R.DATA')

    def reset(self):
        self.steps = 0
        self.episode_counter += 1  # Increment episode count

        if int(self.env_id) in [5, 6, 7, 8]:  # Testing case, use TRUE_DECK
            source_file1 = 'realizations/TRUE_DECK0R.DATA'
            source_file2 = 'realizations/TRUE_DECK0.DATA'
        else:  # Training case, select a random SIM_DECK file
            sim_deck_idx = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            source_file1 = f'realizations/SIM_DECK{sim_deck_idx}0R.DATA'
            source_file2 = f'realizations/SIM_DECK{sim_deck_idx}0.DATA'
        dest_folder = f'{self.env_id}'
        dest_file1 = os.path.join(dest_folder, 'TRUE_DECK0R.DATA')
        dest_file2 = os.path.join(dest_folder, f'TRUE_DECK{self.env_id}_0.DATA')

        # Copy and rename the files
        shutil.copy(source_file1, dest_file1)
        shutil.copy(source_file2, dest_file2)
        self.deck = self.parser.parse(f'{self.env_id}/TRUE_DECK0R.DATA')
        
        # Ensure the destination folder exists
        os.makedirs(dest_folder, exist_ok=True)


        for item in os.listdir(dest_folder):
            item_path = os.path.join(dest_folder, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Error deleting subfolder {item_path}: {e}")        


        self.run_simulation(f'{self.env_id}/TRUE_DECK{self.env_id}_0.DATA', 0)
        obs = self.get_observation(0)
        return obs , {}

    def step(self, action):
        print(f"Received action: {action}")
        self.steps += 1
        unnormalized_action = self.action_low + action * (self.action_high - self.action_low)

        # Apply the sum constraint on gas injections

        self.modify_deck(unnormalized_action, self.steps)
        try:
            self.run_simulation(f'{self.env_id}/TRUE_DECK{self.env_id}_{self.steps}.DATA', self.steps)
            obs = self.get_observation(self.steps)
            reward,penalized_reward = self.compute_reward(obs,self.steps)
            self.raw_reward_log.append({"environment":self.env_id,"episode": self.episode_counter, "step": self.steps, "raw_reward": reward})
            done = self.steps >= self.max_steps
        except Exception as e:  # Catch any exceptions that might occur during the simulation
            print(f"Error during simulation: {e}")
            obs = self.observation_space.sample()  # Sample a random observation 
            reward, penalized_reward = 0,0  # Assign a large negative reward to penalize failed simulations
            done = True
        if done:
            self.save_log()
        dest_folder = f'{self.env_id}'

        for item in os.listdir(dest_folder):
            item_path = os.path.join(dest_folder, item)

            if os.path.isdir(item_path):
                try:
                    folder_step = int(item)
                    
                    if folder_step < self.steps - 3:
                        shutil.rmtree(item_path)
                except ValueError:
                    pass
            if os.path.isfile(item_path):
                try:
                    # Extract step number from filename pattern TRUE_DECK{self.env_id}_{file_step}.*
                    file_name = os.path.basename(item)
                    if file_name.startswith(f"TRUE_DECK{self.env_id}_"):
                        file_step = int(file_name.split(f"TRUE_DECK{self.env_id}_")[1].split('.')[0])

                        # Remove files where file_step < current_step - 3
                        if file_step < self.steps - 3:
                            os.remove(item_path)
                except (ValueError, IndexError):
                    pass

        return obs, penalized_reward, done,False, {}

    def render(self, mode='human'):
        pass

    def modify_deck(self, action, step):
        target_line_number = 162358
        new_content = f"""
        '{step-1}/TRUE_DECK{self.env_id}_{step-1}' {step} 1* 1*/ '
        """

        # Split the string into lines
        lines = str(self.deck).split('\n')
        lines[target_line_number] = new_content

        modified_str = '\n'.join(lines)
        self.deck = self.parser.parse_string(modified_str)

        prod_rate1, prod_rate2, prod_rate3 = action[0:3]
        gas_inj1, gas_inj2, gas_inj3 = action[3:6]

        gconinje_string = f'''
                WCONINJE
                'I1' 'GAS' 'OPEN' 'RATE' {gas_inj1}E3 1* 9500 /
                'I2' 'GAS' 'OPEN' 'RATE' {gas_inj2}E3 1* 9500 /
                'I3' 'GAS' 'OPEN' 'RATE' {gas_inj3}E3 1* 9500 /
                /
                 '''
        self.deck = self.parser.parse_string(str(self.deck)+gconinje_string)

        wconprod_string = f"""
        WCONPROD
        'P1' 'OPEN' 'ORAT' {prod_rate1}E3 1* 6E3 2* 2500 /
        'P2' 'OPEN' 'ORAT' {prod_rate2}E3 1* 6E3 2* 2500 /
        'P3' 'OPEN' 'ORAT' {prod_rate3}E3 1* 6E3 2* 2500 /

        /"""

        tstep_string = """
        TSTEP\n
        180\n
        /
        """
        self.deck = self.parser.parse_string(str(self.deck)+wconprod_string+tstep_string)
        with open(f'{self.env_id}/TRUE_DECK{self.env_id}_{step}.DATA', 'w') as file:
            file.write(str(self.deck))

    def run_simulation(self, deck_filename, step):
        subprocess.run(["flow", deck_filename, f"--output-dir=./{self.env_id}/{step}"])

    def get_observation(self, step):
        summary = ESmry(f'{self.env_id}/{step}/TRUE_DECK{self.env_id}_{step}.SMSPEC')
        rst = ERst(f'{self.env_id}/{step}/TRUE_DECK{self.env_id}_{step}.UNRST')
        init = EclFile(f'{self.env_id}/{step}/TRUE_DECK{self.env_id}_{step}.INIT')
        porv = init['PORV']
        active = porv > 1e-20

        report_step = rst.report_steps[-1]
        sw = np.zeros_like(porv)
        pressure = np.zeros_like(porv)
        sw[active] = 1- rst['SGAS', report_step]
        pressure[active] = rst['PRESSURE', report_step]

        
        

        water_sat_obs = sw.reshape(1,4, 163, 120) # Gas saturation at the last time step reshaped to the grid size
        pressure_obs = pressure.reshape(1,4, 163, 120) # Pressure at the last time step reshaped to the grid size
        pressure_obs = pressure_obs/11000
        three_d_image = np.concatenate((pressure_obs, water_sat_obs), axis=0)

        if step == 0:
            well_observations = np.array([summary['WGIR:I1'][0], summary['WGIR:I2'][0], summary['WGIR:I3'][0], summary['WGPR:P1'][0], summary['WGPR:P2'][0], summary['WGPR:P3'][0], summary['WOPR:P1'][0], summary['WOPR:P2'][0], summary['WOPR:P3'][0], summary['WBHP:I1'][0], summary['WBHP:I2'][0], summary['WBHP:I3'][0], summary['WBHP:P1'][0], summary['WBHP:P2'][0], summary['WBHP:P3'][0]]).flatten()
            well_observations_90 = np.zeros((180, 15))
    
    # Assign the initial well_observations to the first row
            well_observations_90[-1, :] = well_observations    
        else:
            t = np.array([summary['TIME']]).flatten()
            t_int = np.round(t).astype(int)
            t_modified = np.insert(t_int, 0, t_int[-1] - 180)
            intervals = np.diff(t_modified)
            keys = ['WGIR:I1', 'WGIR:I2', 'WGIR:I3', 'WGPR:P1', 'WGPR:P2', 'WGPR:P3', 'WOPR:P1', 'WOPR:P2', 'WOPR:P3', 'WBHP:I1', 'WBHP:I2', 'WBHP:I3', 'WBHP:P1', 'WBHP:P2', 'WBHP:P3']

            repeated_arrays = []
            for key in keys:
                repeated_array = np.repeat(np.array(summary[key]).flatten(), intervals)
                repeated_arrays.append(repeated_array)

            # Concatenate the arrays along the second axis (column-wise)
            well_observations_90 = np.column_stack(repeated_arrays)
        

        well_observations_90 = well_observations_90/100000
        selected_indices = np.linspace(0, 179, 9, endpoint=True, dtype=int)
        well_observations_selected = well_observations_90[selected_indices, :]


        return {
            "res_state": three_d_image,
            "well_observations": well_observations_selected}

    def compute_reward(self, obs,step):
        summary = ESmry(f'{self.env_id}/{step}/TRUE_DECK{self.env_id}_{step}.SMSPEC')
        if step ==0:
            t=1
            FWPR = np.array(summary['FOPR']).flatten()
            FGPR = np.array(summary['FGPR']).flatten()
            FGIR = np.array(summary['FGIR']).flatten()
            days = 1 
        else:
            t = np.array([summary['TIME']]).flatten()
            t_int = np.round(t).astype(int)
            t_modified = np.insert(t_int, 0, t_int[-1] - 180)
            intervals = np.diff(t_modified)

            FWPR = np.repeat(np.array(summary['FOPR']).flatten(), intervals)
            FGPR = np.repeat(np.array(summary['FGPR']).flatten(), intervals)
            FGIR = np.repeat(np.array(summary['FGIR']).flatten(), intervals)

            days = np.arange(t[-1]-180+1,t[-1]+1)

        co2_inflow = 30 * 0.0019 / 0.035 # 30 euro/tonne * tonne/m^3 * m^3/Mscf
        co2_outflow = 6.2 * 0.0019 / 0.035 # 6.2 euro/tonne ...
        brine_treat = 4.83 * 1.2 * 0.159 # 4.83 euro/tonne * tonne/m^3 * m^3/stb

        discount_rate = 0.0142  # 1.42%

        # Convert annual inflation rate to daily inflation rate
        daily_discount_rate = (1 + discount_rate) ** (1 / 365) - 1

        # Calculate daily cash flows
        co2_inflow_cash_flows = (FGIR - FGPR) * co2_inflow
        co2_outflow_cash_flows = FGIR * co2_outflow
        brine_treatment_cash_flows = FWPR * brine_treat

        # Apply inflation adjustment to cash flows
        co2_inflow_present_values = co2_inflow_cash_flows/ (1 + daily_discount_rate) ** days
        co2_outflow_present_values = co2_outflow_cash_flows / (1 + daily_discount_rate) ** days
        brine_treatment_present_values = brine_treatment_cash_flows / (1 + daily_discount_rate) ** days

        # Calculate NPV
        reward = (np.sum(co2_inflow_present_values) - np.sum(co2_outflow_present_values) - np.sum(brine_treatment_present_values))/1e06
        penalized_reward = 0
        #press = np.mean(summary['FPR'])
        #penalized_reward -= np.abs(press-6000)/100

        avg_inj_rate =  np.mean(summary['FGIR'])
        if (avg_inj_rate<220000) and (avg_inj_rate>160000):
            penalized_reward +=20
        elif avg_inj_rate>100000:
            penalized_reward +=10
        elif avg_inj_rate<40000:
            penalized_reward-=30
        else:
            penalized_reward -=20
        return reward,penalized_reward
    
    def save_log(self):
        os.makedirs('rew', exist_ok=True)

        log_filename = f'{self.env_id}_reward_log.json'
        with open(log_filename, 'w') as f:
            json.dump(self.raw_reward_log, f, indent=4)

if __name__ == "__main__":
    env = ReservoirEnv(env_id="TEST_ENV")
    obs, _ = env.reset()
    print("Initial Observation:", obs)

    for _ in range(5):  # Run for a few steps
        action = env.action_space.sample()  # Sample a random action
        obs, reward, done, bool, _ = env.step(action)
        print("Observation:", obs)
        print("Reward:", reward)
        print("Done:", done)
        if done:
            break