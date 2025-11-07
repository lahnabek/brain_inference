"""Run remotly sbi_simulation.py on milan cluster and sync save files to local machine."""


import paramiko
import os
import stat
import getpass
import select
import time

from params import NJOBS
hostname = "topaze.ccc.cea.fr"
username = "bekrilah"

remote_base_dir = "/ccc/scratch/cont003/drf/bekrilah/tvb_model_reference"
local_base_dir = "/volatile/home/lb283126/Documents/Code/tvb_model_reference"

remote_sim_dir = remote_base_dir + "/simulation_file"
local_sim_dir = os.path.join(local_base_dir, "simulation_file")

remote_not_dir = remote_base_dir + "/notebooks"
local_not_dir = os.path.join(local_base_dir, "notebooks")

remote_inf_dir = remote_base_dir + "/inference"
local_inf_dir = os.path.join(local_base_dir, "inference")

remote_save_dir = remote_base_dir + "/save_file"
local_save_dir = os.path.join(local_base_dir, "save_file")

remote_script = remote_inf_dir + "/sbi_simulation.py"





import posixpath

def sync_dir(sftp, remote_dir, local_dir, depth=0):
    """Synchronise récursivement le dossier distant remote_dir dans local_dir.
    Affiche le nombre total d’éléments rencontrés."""

    transferred = 0
    total = 0

    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    entries = sftp.listdir_attr(remote_dir)
    total += len(entries)

    for entry in entries:
        remote_path = posixpath.join(remote_dir, entry.filename)
        local_path = os.path.join(local_dir, entry.filename)

        if stat.S_ISDIR(entry.st_mode):
            t, sub_total = sync_dir(sftp, remote_path, local_path, depth + 1)
            transferred += t
            total += sub_total
        else:
            if not os.path.exists(local_path):
                sftp.get(remote_path, local_path)
                transferred += 1

    if depth == 0:
        print(f"{total} élément(s) trouvé(s) dans le dossier distant '{remote_dir}'")
        print(f"{transferred} fichier(s) transféré(s).")

    return transferred, total


def upload_dir(sftp, local_dir, remote_dir, force=True):
    """Upload récursivement un dossier local vers le dossier distant.
    Si force=True, on écrase toujours les fichiers existants."""
    transferred = 0
    try:
        sftp.listdir(remote_dir)
    except IOError:
        sftp.mkdir(remote_dir)

    for item in os.listdir(local_dir):
        local_path = os.path.join(local_dir, item)
        remote_path = remote_dir + "/" + item

        if os.path.isdir(local_path):
            transferred += upload_dir(sftp, local_path, remote_path, force)
        else:
            try:
                sftp.stat(remote_path)
                if force:
                    sftp.put(local_path, remote_path)
                    transferred += 1
            except IOError:
                sftp.put(local_path, remote_path)
                transferred += 1
    return transferred

import select
import sys

def run_remote_script(ssh):
    remote_script = remote_base_dir + "/inference/sbi_simulation.py"
    activate_env = "source /ccc/scratch/cont003/drf/bekrilah/monenv/bin/activate"
    load_module = "module load python3/3.10.6"
    cd_command = f"cd {remote_base_dir}"

    # Modifie ici avec ta commande de soumission qui réserve les 120 cœurs (exemple avec srun)
    command = (
        f"/bin/bash -l -c '{load_module} && {activate_env} && {cd_command} && "
        f"srun --partition=milan --cpus-per-task=100 python {remote_script}'"
    )

    print(f"Lancement du script distant avec :\n{command}\n")

    # Le reste du code reste inchangé, qui exécute la commande et affiche stdout/stderr
    try:
        stdin, stdout, stderr = ssh.exec_command(command, get_pty=True)
        stdout_chan = stdout.channel
        stderr_chan = stderr.channel

        while not stdout_chan.exit_status_ready():
            rl, wl, xl = select.select([stdout_chan, stderr_chan], [], [], 1.0)
            for chan in rl:
                if chan.recv_ready():
                    data = chan.recv(1024).decode()
                    if chan == stdout_chan:
                        print(data, end="")
                    else:
                        print(data, end="", file=sys.stderr)

        while stdout_chan.recv_ready():
            data = stdout_chan.recv(1024).decode()
            print(data, end="")

        while stderr_chan.recv_ready():
            data = stderr_chan.recv(1024).decode()
            print(data, end="", file=sys.stderr)

        exit_status = stdout_chan.recv_exit_status()
        print(f"\nStatut de sortie : {exit_status}")

    except Exception as e:
        print(f"Erreur lors de l'exécution du script distant : {e}")






def sync_files(force_sync=True):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    password = getpass.getpass(f"Mot de passe SSH pour {username}@{hostname} : ")

    try:
        ssh.connect(hostname, username=username, password=password)
        sftp = ssh.open_sftp()

        if force_sync:
            nb_sim = upload_dir(sftp, local_sim_dir, remote_sim_dir, force=True)
            nb_inf = upload_dir(sftp, local_inf_dir, remote_inf_dir, force=True)
            nb_not = upload_dir(sftp, local_not_dir, remote_not_dir, force=True)
        else:
            nb_sim = upload_dir(sftp, local_sim_dir, remote_sim_dir, force=False)
            nb_inf = upload_dir(sftp, local_inf_dir, remote_inf_dir, force=False)
            nb_not = upload_dir(sftp, local_not_dir, remote_not_dir, force=False)

        print(f"Upload simulation_file : {nb_sim} fichiers transférés.")
        print(f"Upload inference : {nb_inf} fichiers transférés.")
        print(f"Upload notebooks : {nb_not} fichiers transférés.")

        # === Création automatique du script sbatch ===
        remote_script_path = os.path.join(remote_base_dir, "inference/sbi_simulation.py")
        sbatch_script_path = os.path.join(remote_base_dir, "launch_sbi_sim.sh")
        remote_logs_dir = os.path.join(remote_base_dir, "logs")

        sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=sbi_sim
#SBATCH --output={remote_logs_dir}/sbi_sim_%j.out
#SBATCH --error={remote_logs_dir}/sbi_sim_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --time=01:00:00
#SBATCH --partition=milan

module load python3/3.10.6
source /ccc/scratch/cont003/drf/{username}/monenv/bin/activate
cd {remote_base_dir}
python {remote_script_path}
"""

        # Création du dossier logs s'il n'existe pas
        try:
            sftp.stat(remote_logs_dir)
        except FileNotFoundError:
            sftp.mkdir(remote_logs_dir)

        # Upload du script sbatch
        with sftp.open(sbatch_script_path, "w") as f:
            f.write(sbatch_script)
        sftp.chmod(sbatch_script_path, 0o755)
        print(f"Script SLURM {sbatch_script_path} créé et transféré.")

        # === Soumission du job ===
        submit_cmd = f"cd {remote_base_dir} && sbatch launch_sbi_sim.sh"
        stdin, stdout, stderr = ssh.exec_command(submit_cmd)
        job_info = stdout.read().decode().strip()
        print("Job soumis :", job_info)

        # Récupération du job ID pour suivre le fichier log
        job_id = None
        import re
        match = re.search(r'Submitted batch job (\d+)', job_info)
        if match:
            job_id = match.group(1)


        

        if job_id:
            log_file = f"{remote_logs_dir}/sbi_sim_{job_id}.out"
            print(f"Attente de la création du fichier log : {log_file}")

            # Attendre que le fichier log apparaisse (timeout de 2 minutes max)
            max_wait = 20*60  # secondes
            wait_time = 0
            while True:
                try:
                    sftp.stat(log_file)
                    break
                except FileNotFoundError:
                    time.sleep(2)
                    wait_time += 2
                    if wait_time >= max_wait:
                        print(" Timeout : fichier log toujours inexistant.")
                        return

            print(f"🎬 Affichage en direct de : {log_file}")
            tail_cmd = f"tail -f {log_file}"
            stdin, stdout, stderr = ssh.exec_command(tail_cmd, get_pty=True)
            
            try:
                while True:
                    
                    rl, wl, xl = select.select([stdout.channel], [], [], 1.0)
                    if stdout.channel.recv_ready():
                        output = stdout.channel.recv(1024).decode()
                        print(output, end="")
            except KeyboardInterrupt:
                print("\n Suivi interrompu par l'utilisateur.")

        else:
            print("Impossible de récupérer le Job ID pour le suivi du fichier log.")

        # === Téléchargement des résultats ===
        print("Téléchargement des fichiers de résultats...")
        nb_save = sync_dir(sftp, remote_save_dir, local_save_dir)
        print(f"Téléchargement save_file : {nb_save} fichiers transférés.")

        sftp.close()
    finally:
        ssh.close()



def upload_to_scratch(username, hostname, local_base_dir):
    remote_base_dir = f"/ccc/scratch/cont003/drf/{username}/tvb_model_reference"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    password = getpass.getpass(f"Mot de passe SSH pour {username}@{hostname} : ")

    try:
        ssh.connect(hostname, username=username, password=password)
        sftp = ssh.open_sftp()

        # Crée un dossier distant récursivement (comme mkdir -p)
        def mkdir_p(remote_directory):
            dirs = remote_directory.strip("/").split("/")
            path = ""
            for dir in dirs:
                path += "/" + dir
                try:
                    sftp.stat(path)
                except FileNotFoundError:
                    print(f"[mkdir] Création de : {path}")
                    sftp.mkdir(path)

        # Création du dossier base s'il n'existe pas
        try:
            sftp.stat(remote_base_dir)
            print(f"[Info] Le dossier {remote_base_dir} existe déjà.")
        except FileNotFoundError:
            print(f"[Info] Création du dossier {remote_base_dir}")
            mkdir_p(remote_base_dir)

        # Dossiers à exclure du transfert
        exclude_dirs = {"original_dataset", "test", "save_file", "tvb-gdist", "sbi-logs"}

        # Fonction récursive d'upload
        def upload_dir(local_dir, remote_dir):
            for item in os.listdir(local_dir):
                if item in exclude_dirs:
                    print(f"[Exclu] Dossier ignoré : {item}")
                    continue

                local_path = os.path.join(local_dir, item)
                remote_path = remote_dir + "/" + item

                if os.path.isdir(local_path):
                    try:
                        sftp.stat(remote_path)
                        print(f"[Info] Dossier déjà présent : {remote_path}")
                    except FileNotFoundError:
                        print(f"[mkdir] Création de dossier : {remote_path}")
                        sftp.mkdir(remote_path)
                    upload_dir(local_path, remote_path)
                else:
                    print(f"[Upload] {local_path} → {remote_path}")
                    sftp.put(local_path, remote_path)

        upload_dir(local_base_dir, remote_base_dir)

        print("\n✅ Transfert terminé.")
        sftp.close()
    finally:
        ssh.close()



if __name__ == "__main__":

    # Change la valeur ici pour forcer la synchro ou non
    sync_files(force_sync=True)
    #upload_to_scratch(username, hostname, local_base_dir)
