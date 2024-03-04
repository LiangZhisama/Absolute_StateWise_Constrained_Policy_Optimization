import os
import subprocess
import re
import time
import os.path as osp

def get_gpu_memory(device_id):
    try:
        cmd = f'nvidia-smi --id={device_id} --query-gpu=memory.free --format=csv,noheader'
        result = subprocess.check_output(cmd, shell=True)
        memory_free = int(re.findall(r'\d+', result.decode())[0])
        return memory_free
    except Exception as e:
        print(f"Error getting GPU memory info for device {device_id}: {str(e)}")
        return 0

def get_all_gpu_memory():
    try:
        cmd = 'nvidia-smi --query-gpu=count --format=csv,noheader'
        result = subprocess.check_output(cmd, shell=True)
        gpu_count = int(re.findall(r'\d+', result.decode())[0])
        gpu_memory = [get_gpu_memory(i) for i in range(gpu_count)]
        return gpu_memory
    except Exception as e:
        print(f"Error getting GPU memory info: {str(e)}")
        return []
    
def run(python_files_and_args, available_devices):
    indexed_python_files_and_args = [(index, tup) for index, tup in enumerate(python_files_and_args)]
    
    gpu_not_enough = False
    processes = []
    for index, (file_path, arguments) in enumerate(python_files_and_args):
        gpu_memory = get_all_gpu_memory()
        gpu_memory = [gpu_memory[i] for i in available_devices]
        assert gpu_memory != []
        if max(gpu_memory) < 1300:
            gpu_not_enough = True
            break
        
        best_gpu = available_devices[gpu_memory.index(max(gpu_memory))]
        # vctrpo.py will use the third gpu set by os.environ
        devices = str(best_gpu)
        
        try:
            processes.append(subprocess.Popen(f"CUDA_VISIBLE_DEVICES={devices} python {file_path} {arguments}", shell=True, 
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=False))
            print(f"Task {index} successfully executed on cuda:{best_gpu} with params [{file_path} {arguments}]")
        except Exception as e:
            print(f"Error when starting process {index}: {str(e)}")
        if 'plot' not in file_path:
            time.sleep(10)
    
    print("----------------------------")

    if not gpu_not_enough:
        for process, (index, (file_path, arguments)) in zip(processes, indexed_python_files_and_args):
            if process is not None:
                process.wait()
                if process.returncode == 0:
                    print(f"Task {index} executed successfully with arguments: [{arguments}]")
                else:
                    print(f"Task {index} encountered an error with arguments: [{arguments}]")
    else:
        print("GPU memory is nou enough. Please release the memory manually!")
        # for index, (file_path, arguments) in enumerate(python_files_and_args):
        #     exit_code = os.system(f"pkill -f {arguments}")
        #     if exit_code == 0 :
        #         print(f"Task {index} successfully killed.")
        #     else:
        #         print(f"Kill task {index} failed. Error[{exit_code}].")        

if __name__ == "__main__":
    python_files_and_args = []

    # for task in ['HandManipulateBlockDense']:
    #     for seed in [2]:
    #         for algo in ['trpo', 'ppo', 'a2c', 'vmpo', 'alphappo', 'espo']:
    #             python_files_and_args.append((f"{algo}-robotics/{algo}.py", f"-r {task} --seed {seed}"))
    
    # for task in ['HandReachDense', 'HandManipulateBlockDense', 'HandManipulateEggDense', 'HandManipulatePenDense']:
    #     for seed in [3,4]:
    #         python_files_and_args.append((f"espo-robotics/espo.py", f"-r {task} --seed {seed}"))
    
    # for task in ['HandReachDense', 'HandManipulateEggDense', 'HandManipulatePenDense']:
    #     for seed in [0,1]:
    #         python_files_and_args.append((f"apo-robotics/apo.py", f"-r {task} --seed {seed} --detailed --exp_name apo1 --omega1 0.001 --omega2 0.005"))
    #         python_files_and_args.append((f"apo-robotics/apo.py", f"-r {task} --seed {seed} --detailed --exp_name apo2 --omega1 0.005 --omega2 0.01"))
    #         python_files_and_args.append((f"apo-robotics/apo.py", f"-r {task} --seed {seed} --detailed --exp_name apo3 --omega1 0.01 --omega2 0.005"))
            
    # for task in ['HandManipulateBlockDense']:
    #     for seed in [0,1,2]:
    #         python_files_and_args.append((f"apo-robotics/apo.py", f"-r {task} --seed {seed} --detailed --exp_name apo1 --omega1 0.001 --omega2 0.005"))
    #         python_files_and_args.append((f"apo-robotics/apo.py", f"-r {task} --seed {seed} --detailed --exp_name apo2 --omega1 0.005 --omega2 0.01"))
    #         python_files_and_args.append((f"apo-robotics/apo.py", f"-r {task} --seed {seed} --detailed --exp_name apo3 --omega1 0.01 --omega2 0.005"))


    # for omega1 in [0.001, 0.003, 0.005, 0.007, 0.1]:
    #     for omega2 in [0.001, 0.003, 0.005, 0.007, 0.1]:
    #         for seed in [0,1]:
    #             python_files_and_args.append((f"apo-robotics/apo.py", f"-r HandManipulateEggDense --seed {seed} --exp_name apo-{omega1}-{omega2} --omega1 {omega1} --omega2 {omega2}"))
    #         for seed in [0,1,2]:
    #             python_files_and_args.append((f"apo-robotics/apo.py", f"-r HandManipulateBlockDense --seed {seed} --exp_name apo-{omega1}-{omega2} --omega1 {omega1} --omega2 {omega2}"))    
           
    #task = 'Defense'
    # for robo in ['Ant', 'Walker', 'Humanoid']:
    #         for seed in [0,1,2]:
    #             python_files_and_args.append((f"apo/apo.py", f"--task {task}_{robo}_0Hazards --seed {seed} --detailed --exp_name apo1 --omega1 0.001 --omega2 0.005"))
    #             python_files_and_args.append((f"apo/apo.py", f"--task {task}_{robo}_0Hazards --seed {seed} --detailed --exp_name apo2 --omega1 0.005 --omega2 0.01"))
    #             python_files_and_args.append((f"apo/apo.py", f"--task {task}_{robo}_0Hazards --seed {seed} --detailed --exp_name apo3 --omega1 0.01 --omega2 0.005"))
    
    # for robo in ['Point', 'Swimmer']:
    #     for omega1 in [0.001, 0.003, 0.005, 0.007, 0.1]:
    #         for omega2 in [0.001, 0.003, 0.005, 0.007, 0.1]:
    #             for seed in [0,1]:
    #                 python_files_and_args.append((f"papo/papo.py", f"--task {task}_{robo}_0Hazards --seed {seed} --detailed --exp_name papo-{omega1}-{omega2} --omega1 {omega1} --omega2 {omega2}"))

    # for omega1 in [0.001, 0.003, 0.005, 0.007, 0.1]:
    #     for omega2 in [0.001, 0.003, 0.005, 0.007, 0.1]:
    #         for k in [7.0]:
    #             for seed in [0,1]:
    #                 python_files_and_args.append((f"ascpo/ascpo.py", f"--task Goal_Point_8Hazards_noconti --exp_name ascpo-{omega1}-{omega2}-{k} --omega1 {omega1} --omega2 {omega2} --k {k} --seed {seed}"))
    
    # for robo in ['Point', 'Swimmer']:
    #     for num in [1, 4]:
    #         for seed in [0,1]:
    #             for omega1, omega2, k in [(0.001,0.003,7.0), (0.005,0.001,7.0), (0.005,0.007,7.0), (0.007,0.005,7.0), (0.007,0.007,7.0)]:
    #                     python_files_and_args.append((f"ascpo/ascpo.py", f"--task Goal_{robo}_{num}Hazards_noconti --seed {seed} --exp_name ascpo-{omega1}-{omega2}-{k} --omega1 {omega1} --omega2 {omega2} --k {k}"))

    # for robo in ['Ant', 'Hopper', 'Humanoid', 'Point', 'Swimmer', 'Walker']:
    #     for seed in [0,1]:
    #         python_files_and_args.append((f"papo/papo.py", f"--task {task}_{robo}_0Hazards --seed {seed} --detailed --exp_name papo1 --omega1 0.001 --omega2 0.001"))
    #         python_files_and_args.append((f"papo/papo.py", f"--task {task}_{robo}_0Hazards --seed {seed} --detailed --exp_name papo2 --omega1 0.003 --omega2 0.01"))
    #         python_files_and_args.append((f"papo/papo.py", f"--task {task}_{robo}_0Hazards --seed {seed} --detailed --exp_name papo3 --omega1 0.005 --omega2 0.001"))
    #         python_files_and_args.append((f"papo/papo.py", f"--task {task}_{robo}_0Hazards --seed {seed} --detailed --exp_name papo4 --omega1 0.007 --omega2 0.005"))

    # for algo in ['trpofac', 'trpoipo', 'trpolag', 'cpo', 'pcpo', 'safelayer', 'usl']:
    #     for seed in [0,1]:
    #         for robo in ['Point', 'Swimmer']:
    #             for num in [1,4]:
    #                 python_files_and_args.append((f"{algo}/{algo}.py", f"--task Goal_{robo}_{num}Hazards_noconti --seed {seed}"))

    # for seed in [0,1]:
    #     for robo in ['Point', 'Swimmer']:
    #         for num in [1,4]:
    #             python_files_and_args.append((f"scpo_total/scpo.py", f"--task Goal_{robo}_{num}Hazards_noconti --exp_name scpo-softplus-sub --seed {seed}"))


    for robo in ['Point', 'Swimmer']:
        for seed in [0,1]:
            for num in [1,4]:
                for omega1, omega2, k in [(0.001,0.003,7.0), (0.005,0.001,7.0), (0.005,0.007,7.0), (0.007,0.005,7.0), (0.007,0.007,7.0)]:
                    python_files_and_args.append((f"ascpo/ascpo.py", f"--task Goal_{robo}_{num}Hazards_noconti --seed {seed} --train_vc_iters 160 --exp_name ascpo-sub-delta-160-{omega1}-{omega2}-{k} --omega1 {omega1} --omega2 {omega2} --k {k}"))
                    python_files_and_args.append((f"ascpo/ascpo.py", f"--task Goal_{robo}_{num}Hazards_noconti --seed {seed} --train_vc_iters 160 --exp_name ascpo-delta-160-{omega1}-{omega2}-{k} --omega1 {omega1} --omega2 {omega2} --k {k}"))
                    python_files_and_args.append((f"ascpo/ascpo.py", f"--task Goal_{robo}_{num}Hazards_noconti --seed {seed} --exp_name ascpo-relu-sub-{omega1}-{omega2}-{k} --omega1 {omega1} --omega2 {omega2} --k {k}"))
    
    # for seed in [0,1]:
    #     for robo in ['Point']:
    #         python_files_and_args.append((f"scpo_total/scpo.py", f"--task Goal_{robo}_8Hazards_noconti --seed {seed} --train_vc_iters 160 --model_save --exp_name scpo-softplus-sub-delta-focus-160"))
    #         python_files_and_args.append((f"scpo_total/scpo.py", f"--task Goal_{robo}_8Hazards_noconti --seed {seed} --train_vc_iters 160 --model_save --exp_name scpo-softplus-delta-focus-160"))
    #         python_files_and_args.append((f"scpo_total/scpo.py", f"--task Goal_{robo}_8Hazards_noconti --seed {seed} --model_save --exp_name scpo-relu-sub-focus"))
    #         python_files_and_args.append((f"scpo_total/scpo.py", f"--task Goal_{robo}_8Hazards_noconti --seed {seed} --model_save --exp_name scpo-softplus-sub-focus"))
    #         python_files_and_args.append((f"cpo/cpo.py", f"--task Goal_{robo}_8Hazards_noconti --seed {seed}"))

    # # papo mujoco+robotics group
    # python_files_and_args.append((f"papo-mujoco/papo.py", f"--model_save --epochs 100 -m HumanoidStandup --seed 2 --omega1 0.01 --omega2 0.01"))
    # python_files_and_args.append((f"ppo-mujoco/ppo.py", f"--model_save --epochs 100 -m HumanoidStandup --seed 0"))

    # for seed in [0,1,2,3,4,5,6,7,8,9]:
    #     python_files_and_args.append((f"espapo-robotics/espapo.py", f"--model_save --epochs 200 -r HandReachDense --seed {seed} --omega1 0.1 --omega2 0.003 --detailed"))
    #     python_files_and_args.append((f"ppo-robotics/ppo.py", f"--model_save --epochs 200 -r HandReachDense --seed {seed}"))

    #     python_files_and_args.append((f"espapo-robotics/espapo.py", f"--model_save --epochs 200 -r HandManipulateEggDense --seed {seed} --omega1 0.001 --omega2 0.001 --detailed"))
    #     python_files_and_args.append((f"ppo-robotics/ppo.py", f"--model_save --epochs 200 -r HandManipulateEggDense --seed {seed}"))

    # # papo guard group
    # python_files_and_args.append((f"papo/papo.py", f"--model_save --epochs 120 --task Goal_Ant_0Hazards --seed 1 --omega1 0.001 --omega2 0.001 --detailed"))
    # python_files_and_args.append((f"ppo/ppo.py", f"--model_save --epochs 120 --task Goal_Ant_0Hazards --seed 1"))
    
    # python_files_and_args.append((f"papo/papo.py", f"--model_save --epochs 120 --task Chase_Ant_0Hazards --seed 0 --omega1 0.001 --omega2 0.001 --detailed"))
    # python_files_and_args.append((f"ppo/ppo.py", f"--model_save --epochs 120 --task Chase_Ant_0Hazards --seed 0"))
    
    # python_files_and_args.append((f"papo/papo.py", f"--model_save --epochs 100 --task Goal_Walker_0Hazards --seed 0 --omega1 0.003 --omega2 0.007 --detailed"))
    # python_files_and_args.append((f"ppo/ppo.py", f"--model_save --epochs 100 --task Goal_Walker_0Hazards --seed 1"))
    
    # python_files_and_args.append((f"papo/papo.py", f"--model_save --epochs 100 --task Chase_Walker_0Hazards --seed 1 --omega1 0.003 --omega2 0.01 --detailed"))
    # python_files_and_args.append((f"ppo/ppo.py", f"--model_save --epochs 100 --task Chase_Walker_0Hazards --seed 0"))

    # for seed in [0,1,2]:
    #     python_files_and_args.append((f"papo/papo.py", f"--model_save --epochs 60 --task Goal_Arm3_0Hazards --seed {seed} --omega1 0.003 --omega2 0.01 --detailed"))
    #     python_files_and_args.append((f"ppo/ppo.py", f"--model_save --epochs 60 --task Goal_Arm3_0Hazards --seed {seed}"))

    # python_files_and_args.append((f"papo/papo.py", f"--model_save --epochs 100 --task Chase_Hopper_0Hazards --seed 1 --omega1 0.003 --omega2 0.01 --detailed"))
    # python_files_and_args.append((f"ppo/ppo.py", f"--model_save --epochs 100 --task Chase_Hopper_0Hazards --seed 0"))

    # # apo push group
    # python_files_and_args.append((f"apo/apo.py", f"--model_save --epochs 200 --task Push_Point_0Hazards --seed 1 --omega1 0.01 --omega2 0.005 --detailed"))
    # python_files_and_args.append((f"ppo/ppo.py", f"--model_save --epochs 200 --task Push_Point_0Hazards --seed 2"))
    
    # python_files_and_args.append((f"apo/apo.py", f"--model_save --epochs 200 --task Push_Swimmer_0Hazards --seed 0 --omega1 0.001 --omega2 0.005 --detailed"))
    # python_files_and_args.append((f"ppo/ppo.py", f"--model_save --epochs 200 --task Push_Swimmer_0Hazards --seed 0"))

    # python_files_and_args.append((f"apo/apo.py", f"--model_save --epochs 200 --task Push_Hopper_0Hazards --seed 0 --omega1 0.005 --omega2 0.01 --detailed"))
    # python_files_and_args.append((f"ppo/ppo.py", f"--model_save --epochs 200 --task Push_Hopper_0Hazards --seed 0"))

    # python_files_and_args.append((f"apo/apo.py", f"--model_save --epochs 200 --task Push_Ant_0Hazards --seed 1 --omega1 0.01 --omega2 0.005 --detailed"))
    # python_files_and_args.append((f"ppo/ppo.py", f"--model_save --epochs 200 --task Push_Ant_0Hazards --seed 2"))

    # python_files_and_args.append((f"apo/apo.py", f"--model_save --epochs 200 --task Push_Walker_0Hazards --seed 1 --omega1 0.01 --omega2 0.005 --detailed"))
    # python_files_and_args.append((f"ppo/ppo.py", f"--model_save --epochs 200 --task Push_Walker_0Hazards --seed 2"))

    # apo atari
    # _81 = "--omega1 0.01 --omega2 0.005"
    # _81new = "--omega1 0.01 --omega2 0.005 --detailed"
    # _101 = "--omega1 0.001 --omega2 0.005"
    # _101new = "--omega1 0.001 --omega2 0.005 --detailed"
    # for seed in [0,1,2]:
    #     python_files_and_args.append((f"apo/apo.py", f"--model_save --epochs 500 --save_freq 50 -a AirRaid -s {seed} {_81}"))
    #     python_files_and_args.append((f"apo/apo.py", f"--model_save --epochs 500 --save_freq 50 -a Kangaroo -s {seed} {_101}"))
    #     python_files_and_args.append((f"apo/apo.py", f"--model_save --epochs 500 --save_freq 50 -a Riverraid -s {seed} {_101}"))
    #     python_files_and_args.append((f"apo/apo.py", f"--model_save --epochs 500 --save_freq 50 -a FishingDerby -s {seed} {_101new}"))
    #     python_files_and_args.append((f"apo/apo.py", f"--model_save --epochs 500 --save_freq 50 -a Pong -s {seed} {_101new}"))
    #     python_files_and_args.append((f"apo/apo.py", f"--model_save --epochs 500 --save_freq 50 -a Jamesbond -s {seed} {_101}"))
    #     python_files_and_args.append((f"apo/apo.py", f"--model_save --epochs 500 --save_freq 50 -a Centipede -s {seed} {_101}"))
    #     python_files_and_args.append((f"apo/apo.py", f"--model_save --epochs 500 --save_freq 50 -a Amidar -s {seed} {_101}"))
    #     python_files_and_args.append((f"apo/apo.py", f"--model_save --epochs 500 --save_freq 50 -a Krull -s {seed} {_81}"))
    #     python_files_and_args.append((f"apo/apo.py", f"--model_save --epochs 500 --save_freq 50 -a Zaxxon -s {seed} {_101}"))
    #     python_files_and_args.append((f"apo/apo.py", f"--model_save --epochs 500 --save_freq 50 -a PrivateEye -s {seed} {_101}"))
    #     python_files_and_args.append((f"apo/apo.py", f"--model_save --epochs 500 --save_freq 50 -a MontezumaRevenge -s {seed} {_81}"))
    #     python_files_and_args.append((f"apo/apo.py", f"--model_save --epochs 500 --save_freq 10 -a Tennis -s {seed} {_81new}"))
    #     python_files_and_args.append((f"apo/apo.py", f"--model_save --epochs 500 --save_freq 10 -a Venture -s {seed} {_101}"))

    #     for task in ['AirRaid', 'Kangaroo', 'Riverraid', 'FishingDerby', 'Pong', 'Jamesbond', 'Centipede', 'Amidar', 'Krull', 'Zaxxon', 'PrivateEye', 'MontezumaRevenge']:
    #         python_files_and_args.append((f"ppo/ppo.py", f"--model_save --epochs 500 --save_freq 50 -a {task} -s {seed}"))
    #     for task in ['Tennis', 'Venture']:
    #         python_files_and_args.append((f"ppo/ppo.py", f"--model_save --epochs 500 --save_freq 10 -a {task} -s {seed}"))

    # fix
    # for seed in [0,1,2]:
    #     python_files_and_args.append((f"papo-mujoco/papo.py", f"--model_save --epochs 1000 -m HumanoidStandup --seed {seed} --omega1 0.01 --omega2 0.01"))
    #     python_files_and_args.append((f"ppo-mujoco/ppo.py", f"--model_save --epochs 1000 -m HumanoidStandup --seed {seed}"))
    
    # for task in ['Chase_Ant_0Hazards', 'Goal_Ant_0Hazards']:
    #     for seed in [0,1,2]:
    #         python_files_and_args.append((f"papo/papo.py", f"--model_save --task {task} --seed {seed} --exp_name papo1 --omega1 0.001 --omega2 0.001 --detailed"))
    #         python_files_and_args.append((f"papo/papo.py", f"--model_save --task {task} --seed {seed} --exp_name papo2 --omega1 0.003 --omega2 0.01 --detailed"))
    #         python_files_and_args.append((f"papo/papo.py", f"--model_save --task {task} --seed {seed} --exp_name papo3 --omega1 0.005 --omega2 0.001 --detailed"))
    #         python_files_and_args.append((f"papo/papo.py", f"--model_save --task {task} --seed {seed} --exp_name papo4 --omega1 0.007 --omega2 0.005 --detailed"))
    #         python_files_and_args.append((f"ppo/ppo.py", f"--model_save --task {task} --seed {seed}"))

    # for seed in [0,1,2,3,4,5]:
    #     python_files_and_args.append((f"apo/apo.py", f"--model_save --epochs 200 --task Push_Hopper_0Hazards --seed {seed} --omega1 0.005 --omega2 0.01 --detailed"))
    #     python_files_and_args.append((f"ppo/ppo.py", f"--model_save --epochs 200 --task Push_Hopper_0Hazards --seed {seed}"))

    #     python_files_and_args.append((f"apo/apo.py", f"--model_save --epochs 200 --task Push_Ant_0Hazards --seed {seed} --omega1 0.01 --omega2 0.005 --detailed"))
    #     python_files_and_args.append((f"ppo/ppo.py", f"--model_save --epochs 200 --task Push_Ant_0Hazards --seed {seed}"))  

    # log_dict = {}
    # base_dir = "../Final_Results"
    # origin_dirs = os.listdir(base_dir)[0:1]
    # for origin_dir in origin_dirs:
    #     log_dict[origin_dir] = [osp.join(base_dir, origin_dir, single_dir) for single_dir in os.listdir(osp.join(base_dir, origin_dir))]
    
    # for key in log_dict.keys():
    #     for dire in log_dict[key]:
    #         python_files_and_args.append(("utils/plot.py", f"{dire}/ -s 10 --title {key}/{dire.split('/')[-1]} --reward --cost"))
    
    run(python_files_and_args, [0,1,2,3])
