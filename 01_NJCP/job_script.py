import subprocess

def run_command(command):

    try:
        print(f"Executing: {command}")
        result = subprocess.run(command, shell=True, check=True)
        print(f"Command completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {command}: {e}")
        exit(1)

def main():
    
    commands = [
        "python3 GPR_NJCP_d13C_TexH86_20240214_global.py 0",
        "python3 hyperparam_img_glob.py",
        "python3 GPR_NJCP_d13C_TexH86_20240214_regional.py 1",
        "python3 hyperparam_img_reg.py",
        "python3 GPR_NJCP_d13C_TexH86_20240214_local.py 2",
        "python3 hyperparam_img_loc.py",
        "python3 GPR_NJCP_project_local_20240104.py"
        "python3 GPR_NJCP_d13C_img_20240221.py"
    ]

    for command in commands:
        run_command(command)

if __name__ == "__main__":
    main()