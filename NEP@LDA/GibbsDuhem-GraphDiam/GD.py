import os
import shutil
import subprocess
import numpy as np

#constants and conversions
kbeV = 8.6173303e-5
evtoGPa = 160.2176621

#folders and files

def writeinp(temp, press):
    input_file = "run.in"
    with open(input_file, 'w') as f:
        # Writing the contents of the input file
        f.write(f"##########################\n")
        f.write(f"potential nep.txt\n")
        f.write(f"time_step 0.5\n")
        f.write(f"velocity {temp}\n")  
        f.write(f"ensemble npt_scr {temp} {temp} 100 {press} {press} {press}  50. 50. 50. 1000\n")
        f.write(f"run 100000\n")
        f.write(f"ensemble npt_scr {temp} {temp} 100 {press} {press} {press}  50. 50. 50. 1000\n")
        f.write(f"dump_restart 10000\n")
        f.write(f"dump_thermo 100\n")
        f.write(f"run 100000\n")
        f.write(f"##########################\n")
    return

### Thermodynamic Variables & main loop ###
natmG = 960  # n. atoms in the graphite supercell
natmD = 1000 # n. atoms in the diamond supercell
T0 = 4640.
TEnd = 3400.
deltabeta = 0.05
beta0 = 1./kbeV/T0
betaEnd = 1./kbeV/TEnd
beta = beta0
press0 = 10.3
press = press0

while beta < betaEnd:
    temp = int(1./beta/kbeV)

    # Write input file 'run.in'
    err = writeinp(temp,press)

    # Create directories
    graphite_folder = f"graphite{temp}"
    diamond_folder = f"diamond{temp}"
    os.makedirs(graphite_folder, exist_ok=True)
    os.makedirs(diamond_folder, exist_ok=True)

    # Copy model files with renamed versions
    shutil.copy("graphite.xyz", os.path.join(graphite_folder, "model.xyz"))
    shutil.copy("diamond.xyz", os.path.join(diamond_folder, "model.xyz"))
 
    #Copy run.in and nep.txt
    for folder in [graphite_folder, diamond_folder]:
        shutil.copy("nep.txt", os.path.join(folder, "nep.txt"))
        shutil.copy("run.in", os.path.join(folder, "run.in"))

    print("Running MD: Temperature (K) = ", temp, "Pressure (GPa) = ", press) 

    #Run MD simulations
    graphite_process = subprocess.Popen(f"export CUDA_VISIBLE_DEVICES=0 && cd {graphite_folder} && gpumd3.6 > output.log 2>&1", shell=True)
    diamond_process = subprocess.Popen(f"export CUDA_VISIBLE_DEVICES=1 && cd {diamond_folder} && gpumd3.6 > output.log 2>&1", shell=True)

    graphite_process.wait()
    diamond_process.wait()

    restart_file = os.path.join(graphite_folder, "restart.xyz") 
    shutil.copy(restart_file, "graphite.xyz") 
    restart_file = os.path.join(diamond_folder, "restart.xyz") 
    shutil.copy(restart_file, "diamond.xyz") 

    #Read the enthalpy and the volume from the second half of the file
    thermoG = np.loadtxt(os.path.join(graphite_folder, "thermo.out"))
    thermoD = np.loadtxt(os.path.join(diamond_folder, "thermo.out"))

    volumeG = np.average(thermoG[:,9]*thermoG[:,10]*thermoG[:,11])
    volumeD = np.average(thermoD[:,9]*thermoD[:,10]*thermoD[:,11])

    etotG = np.average(thermoG[:,1]+thermoG[:,2])
    etotD = np.average(thermoD[:,1]+thermoD[:,2])
    
    enthalpyG = etotG + press*volumeG/evtoGPa
    enthalpyD = etotD + press*volumeD/evtoGPa

    #Calculate deltaH and deltaVolume and integrate Gibbs-Duhem equation
    DeltaH = enthalpyG/natmG - enthalpyD/natmD
    DeltaVol = volumeG/natmG - volumeD/natmD

    press = press - DeltaH/DeltaVol/beta * deltabeta * evtoGPa 
    beta += deltabeta  # this is the updated inverse temperature

    print("dH, dV = ", DeltaH, DeltaVol)
    print("New pressure", press)

