import gymnasium as gym
from gymnasium import spaces
import numpy as np
import subprocess
import os
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
import torch
RAMDISK_BASE = "/root/clrm/wm_sc2/env_w_dir"  # Where we want our env's working directory
SOURCE_DIR = "/root/clrm/wm_sc2"  # Where TRUE_DECK0R.DATA and TRUE_DECK0.DATA are located
from collections import deque

LEAK_CELLS = {
    0: (49, 33, 4),
    1: (72, 33, 4),
    2: (47, 56, 4),
    3: (69, 50, 4),
    4: (68, 68, 4),
    5: (64, 80, 4),
    6: (69, 100, 4),
    7: (56, 108, 4),
    8: (48, 90, 4),
    9: (47, 85, 4),
    "true": (47, 75, 4),
}

class ReservoirEnv(gym.Env):
    def __init__(self,env_id):
        super(ReservoirEnv, self).__init__()
        self.env_id = env_id
        self.episode_counter = 0

        self.raw_reward_log = []

        print(f"Initializing environment with ID: {self.env_id}")

        # Define action and observation space
        # Actions are the production rates and gas injection rates
        self.action_low = np.array([5, 5, 5, 5, 5, 5, 5, 5, 10, 10, 10], dtype=np.float32)
        self.action_high = np.array([60, 60, 60, 60, 60, 60, 60, 60, 100,100,100], dtype=np.float32)

        # Define normalized action space between 0 and 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(11,), dtype=np.float32)
        self.history_length = 20  # Define how many past observations to keep
        self.history = deque(maxlen=self.history_length)
        self.history.clear() 
        for _ in range(self.history_length):
            self.history.append(np.zeros((9, 30), dtype=np.float32))
              
        # Observations are the pressure and water saturation for all cells,
        # and well results (gas injection rates, production rates, water production rates, and BHP)
        self.observation_space = spaces.Dict({
            "res_state": spaces.Box(low=0, high=np.inf, shape=(2,4, 163, 120), dtype=np.float32),
            "well_observations": spaces.Box(low=0, high=np.inf, shape=(9,30), dtype=np.float32),
            "history": spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.history_length, 9,30), dtype=np.float32
            ),
        })

        self.parser = Parser()

        self.steps = 0
        self.max_steps = 20
        self.working_dir = os.path.join(RAMDISK_BASE, str(self.env_id))
        os.makedirs(self.working_dir, exist_ok=True)
        self.current_sim_deck = None
        self.is_true_env = None

        if int(self.env_id) in [151,161,171,181,191,201]:  # Testing case, use TRUE_DECK
            source_file1 = os.path.join(SOURCE_DIR, "realizations/TRUE_DECK0R.DATA")
            source_file2 = os.path.join(SOURCE_DIR, "realizations/TRUE_DECK0.DATA")
            self.is_true_env = int(self.env_id) 
        else:  # Training case, select a random SIM_DECK file
            sim_deck_idx = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            self.current_sim_deck = sim_deck_idx
            source_file1 = os.path.join(SOURCE_DIR, f'realizations/SIM_DECK{sim_deck_idx}0R.DATA')
            source_file2 = os.path.join(SOURCE_DIR, f'realizations/SIM_DECK{sim_deck_idx}0.DATA')
        dest_file1 = os.path.join(self.working_dir, "TRUE_DECK0R.DATA")
        dest_file2 = os.path.join(self.working_dir, f"TRUE_DECK{self.env_id}_0.DATA")

        shutil.copy(source_file1, dest_file1)
        shutil.copy(source_file2, dest_file2)
        self.deck = self.parser.parse(dest_file1)

    def reset(self):
        self.steps = 0
        self.episode_counter += 1  # Increment episode count
        self.history.clear()
        for _ in range(self.history_length):
            self.history.append(np.zeros((9, 30), dtype=np.float32))
        for item in os.listdir(self.working_dir):
            item_path = os.path.join(self.working_dir, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except Exception as e:
                print(f"Error cleaning up {item_path}: {e}")
        self.is_true_env = None
        self.current_sim_deck = None


        if int(self.env_id) in [151,161,171,181,191,201]:  # Testing case, use TRUE_DECK
            source_file1 = 'realizations/TRUE_DECK0R.DATA'
            source_file2 = 'realizations/TRUE_DECK0.DATA'
            self.is_true_env = int(self.env_id)
        else:  # Training case, select a random SIM_DECK file
            sim_deck_idx = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            source_file1 = f'realizations/SIM_DECK{sim_deck_idx}0R.DATA'
            source_file2 = f'realizations/SIM_DECK{sim_deck_idx}0.DATA'
            self.current_sim_deck = sim_deck_idx
        dest_file1 = os.path.join(self.working_dir, "TRUE_DECK0R.DATA")
        dest_file2 = os.path.join(self.working_dir, f"TRUE_DECK{self.env_id}_0.DATA")

        shutil.copy(source_file1, dest_file1)
        shutil.copy(source_file2, dest_file2)

        self.deck = self.parser.parse(dest_file1)

        # Run the simulation
        self.run_simulation(dest_file2, 0)

        obs = self.get_observation(0)
        obs["history"] = np.array(self.history, dtype=np.float32)
        return obs, {"t_state": 0, "max_steps": self.max_steps}
    
    def step(self, action):
        self.steps += 1
        action = np.clip(action, -1.0, 1.0)
        action_in_0_1 = 0.5 * (action + 1.0)
        unnormalized_action = self.action_low + action_in_0_1 * (self.action_high - self.action_low)

        self.modify_deck(unnormalized_action, self.steps)

        try:
            deck_file = os.path.join(self.working_dir, f"TRUE_DECK{self.env_id}_{self.steps}.DATA")
            self.run_simulation(deck_file, self.steps)
            obs = self.get_observation(self.steps)
            self.history.append(obs["well_observations"])
            obs["history"] = np.array(self.history, dtype=np.float32)
            reward, penalized_reward = self.compute_reward(obs, self.steps)
            self.raw_reward_log.append({
                "environment": self.env_id,
                "episode": self.episode_counter,
                "step": self.steps,
                "raw_reward": reward,
                "pen_reward": penalized_reward
            })
            done = (self.steps >= self.max_steps)
        except Exception as e:
            print(f"Error during simulation: {e}")
            obs = self.observation_space.sample()
            obs["res_state"] = obs["res_state"].astype(np.float32)
            obs["well_observations"] = obs["well_observations"].astype(np.float32)
            obs["history"] = obs["history"].astype(np.float32)

            reward, penalized_reward = 0, 0
            done = True

        if done:
            self.save_log()

        for item in os.listdir(self.working_dir):
            item_path = os.path.join(self.working_dir, item)
            if os.path.isdir(item_path):
                try:
                    folder_step = int(item)
                    if folder_step < self.steps - 3:
                        shutil.rmtree(item_path)
                except ValueError:
                    pass
            else:
                file_name = os.path.basename(item_path)
                if file_name.startswith(f"TRUE_DECK{self.env_id}_"):
                    try:
                        file_step = int(file_name.split(f"TRUE_DECK{self.env_id}_")[1].split('.')[0])
                        if file_step < self.steps - 3:
                            os.remove(item_path)
                    except (ValueError, IndexError):
                        pass

        print(done) 
        info = {"t_state": self.steps - 1, "t_next": self.steps, "max_steps": self.max_steps}
        return obs, penalized_reward, done, False, info

    def render(self, mode='human'):
        pass


    def modify_deck(self, action, step):
        target_line_number = 162361
        new_content = f"""
        '{step-1}/TRUE_DECK{self.env_id}_{step-1}' {step} 1* 1*/ '
        """

        # Split the string into lines
        lines = str(self.deck).split('\n')
        lines[target_line_number] = new_content

        modified_str = '\n'.join(lines)
        self.deck = self.parser.parse_string(modified_str)

        prod_rate1, prod_rate2, prod_rate3, prod_rate4, prod_rate5, prod_rate6, prod_rate7, prod_rate8 = action[0:8]
        gas_inj1, gas_inj2, gas_inj3 = action[8:11]

        gconinje_string = f'''
            GCONINJE
                'FIELD' 'GAS' 'RATE' 300E3 3* 'NO' /
            'Inj' 'GAS' 'RATE' 300E3 3* 'NO' /
                 /
            WCONINJE
            'I1' 'GAS' 'OPEN' 'RATE' {gas_inj1}E3 1* 9000 /
            'I2' 'GAS' 'OPEN' 'RATE' {gas_inj2}E3 1* 9000 /
            'I3' 'GAS' 'OPEN' 'RATE' {gas_inj3}E3 1* 9000 /
            /
        '''
        self.deck = self.parser.parse_string(str(self.deck) + gconinje_string)

        wconprod_string = f"""
        WCONPROD
        'P1' 'OPEN' 'ORAT' {prod_rate1}E3 1* 3E3 2* 3000 /
        'P2' 'OPEN' 'ORAT' {prod_rate2}E3 1* 3E3 2* 3000 /
        'P3' 'OPEN' 'ORAT' {prod_rate3}E3 1* 3E3 2* 3000 /
        'P4' 'OPEN' 'ORAT' {prod_rate4}E3 1* 3E3 2* 3000 /
        'P5' 'OPEN' 'ORAT' {prod_rate5}E3 1* 3E3 2* 3000 /
        'P6' 'OPEN' 'ORAT' {prod_rate6}E3 1* 3E3 2* 3000 /
        'P7' 'OPEN' 'ORAT' {prod_rate7}E3 1* 3E3 2* 3000 /
        'P8' 'OPEN' 'ORAT' {prod_rate8}E3 1* 3E3 2* 3000 /
        /
        """

        tstep_string = """
        TSTEP
        730
        /
        """
        self.deck = self.parser.parse_string(str(self.deck) + wconprod_string + tstep_string)

        deck_file = os.path.join(self.working_dir, f"TRUE_DECK{self.env_id}_{step}.DATA")
        with open(deck_file, 'w') as file:
            file.write(str(self.deck))
    def run_simulation(self, deck_filename, step):
        output_dir = os.path.join(self.working_dir, str(step))
        subprocess.run([
            "flow",
            deck_filename,
            f"--output-dir={output_dir}"
        ],
        stdout=subprocess.DEVNULL,  # <--- disable stdout
        stderr=subprocess.DEVNULL   # <--- disable stderr
        )
    def get_observation(self, step):
        output_dir = os.path.join(self.working_dir, str(step))
        smspec_file = os.path.join(output_dir, f"TRUE_DECK{self.env_id}_{step}.SMSPEC")
        unrst_file = os.path.join(output_dir, f"TRUE_DECK{self.env_id}_{step}.UNRST")
        init_file  = os.path.join(output_dir, f"TRUE_DECK{self.env_id}_{step}.INIT")

        summary = ESmry(smspec_file)
        rst = ERst(unrst_file)
        init = EclFile(init_file)
        porv = init['PORV']
        active = porv > 1e-20

        report_step = rst.report_steps[-1]
        sw = np.zeros_like(porv)
        pressure = np.zeros_like(porv)
        sw[active] = 1 - rst['SGAS', report_step]
        pressure[active] = rst['PRESSURE', report_step]

        water_sat_obs = sw.reshape(1, 4, 163, 120)
        pressure_obs = pressure.reshape(1, 4, 163, 120) / 11000.0
        three_d_image = np.concatenate((pressure_obs, water_sat_obs), axis=0).astype(np.float32)
        # Well observations
        if step == 0:
            injection_observations = [summary[f'WGIR:I{i+1}'][0] for i in range(3)]
            production_observations_gas = [summary[f'WGPR:P{i+1}'][0] for i in range(8)]
            production_observations_oil = [summary[f'WOPR:P{i+1}'][0] for i in range(8)]
            injection_observations_bhp = [summary[f'WBHP:I{i+1}'][0] for i in range(3)]
            production_observations_bhp = [summary[f'WBHP:P{i+1}'][0] for i in range(8)]
            well_observations = np.array(
                injection_observations
                + production_observations_gas
                + production_observations_oil
                + injection_observations_bhp
                + production_observations_bhp
            ).flatten()
            well_observations_yr = np.zeros((730, 30))
            well_observations_yr[-1, :] = well_observations
        else:
            t = np.array([summary['TIME']]).flatten()
            t_int = np.round(t).astype(int)
            t_modified = np.insert(t_int, 0, t_int[-1] - 730)
            intervals = np.diff(t_modified)

            injection_keys = [f'WGIR:I{i+1}' for i in range(3)]
            prod_gas_keys = [f'WGPR:P{i+1}' for i in range(8)]
            prod_oil_keys = [f'WOPR:P{i+1}' for i in range(8)]
            inj_bhp_keys = [f'WBHP:I{i+1}' for i in range(3)]
            prod_bhp_keys = [f'WBHP:P{i+1}' for i in range(8)]
            keys = injection_keys + prod_gas_keys + prod_oil_keys + inj_bhp_keys + prod_bhp_keys

            repeated_arrays = []
            for key in keys:
                repeated_array = np.repeat(np.array(summary[key]).flatten(), intervals)
                repeated_arrays.append(repeated_array)
            well_observations_yr = np.column_stack(repeated_arrays)

        well_observations_yr = well_observations_yr / 100000.0
        if well_observations_yr.shape[0] < 730:
            well_observations_yr = np.zeros((730, 30), dtype=np.float32)

        selected_indices = np.linspace(0, 729, 9, endpoint=True, dtype=int)
        well_observations_selected = well_observations_yr[selected_indices, :].astype(np.float32)

        obs =  {
            "res_state": three_d_image,
            "well_observations": well_observations_selected
        }
        return obs

    def compute_reward(self, obs, step):
        output_dir = os.path.join(self.working_dir, str(step))
        smspec_file = os.path.join(output_dir, f"TRUE_DECK{self.env_id}_{step}.SMSPEC")
        summary = ESmry(smspec_file)
        unrst_file = os.path.join(output_dir, f"TRUE_DECK{self.env_id}_{step}.UNRST")
        init_file  = os.path.join(output_dir, f"TRUE_DECK{self.env_id}_{step}.INIT")
        rst = ERst(unrst_file)
        init = EclFile(init_file)
        porv = init['PORV']
        active = porv > 1e-20
        if step == 0:
            # ...
            t = 1
            FWPR = np.array(summary['FOPR']).flatten()
            FGPR = np.array(summary['FGPR']).flatten()
            FGIR = np.array(summary['FGIR']).flatten()
            days = 1 
        else:
            t = np.array([summary['TIME']]).flatten()
            t_int = np.round(t).astype(int)
            t_modified = np.insert(t_int, 0, t_int[-1] - 730)
            intervals = np.diff(t_modified)

            FWPR = np.repeat(np.array(summary['FOPR']).flatten(), intervals)
            FGPR = np.repeat(np.array(summary['FGPR']).flatten(), intervals)
            FGIR = np.repeat(np.array(summary['FGIR']).flatten(), intervals)
            days = np.arange(t[-1] - 730 + 1, t[-1] + 1)

        co2_inflow = 30 * 0.0019 / 0.035
        co2_outflow = 6.2 * 0.0019 / 0.035
        brine_treat = 4.83 * 1.2 * 0.159
        discount_rate = 0.0142
        daily_discount_rate = (1 + discount_rate) ** (1 / 365) - 1

        co2_inflow_cash_flows = (FGIR - FGPR) * co2_inflow
        co2_outflow_cash_flows = FGIR * co2_outflow
        brine_treatment_cash_flows = FWPR * brine_treat

        co2_inflow_pv = co2_inflow_cash_flows / (1 + daily_discount_rate)**days
        co2_outflow_pv = co2_outflow_cash_flows / (1 + daily_discount_rate)**days
        brine_treat_pv = brine_treatment_cash_flows / (1 + daily_discount_rate)**days

        reward = (np.sum(co2_inflow_pv) - np.sum(co2_outflow_pv) - np.sum(brine_treat_pv)) / 1e08
        penalized_reward = 0
        seq_rate = summary['FGIR'][-1] - summary['FGPR'][-1]

        if ((seq_rate < 220000) and (seq_rate > 170000)):
            penalized_reward = 0.1
        penalized_reward += reward
        sg = np.zeros_like(porv)
        report_step = rst.report_steps[-1]
        sg[active] = rst['SGAS', report_step]
        sat_gas = sg.reshape(4, 163, 120)

        leak_cell_sg = np.max(sat_gas[:,103:123,54:68])
        
        if leak_cell_sg > 0.2:
            penalized_reward -= 10*(leak_cell_sg - 0.2)
        return reward, penalized_reward
    
    def save_log(self):
        os.makedirs('rew', exist_ok=True)
        log_filename = f'rew/{self.env_id}_reward_log.json'
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