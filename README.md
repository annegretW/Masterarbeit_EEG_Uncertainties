## File structure
* data - contains meshes, transfermatrices, electrodes, leadfieldmatrices 
* evaluations - contains notebooks creating evaluations of the results
* python - contains all python code to run the project
* results - contains the results of the algorithm


## How to run the project?
1. Create a config file in 2d/configs.

2. Create Transfermatrix or Leadfieldmatrix (if not already existent).
```
cd 2d/python
python3 create_data.py "LOCAL_CONFIG_PATH"
```

3. Create EEG model.
```
python3 eeg_model.py "LOCAL_CONFIG_PATH"
```

4. Run MCMC algorithm.
```
cd /home/anne/Masterarbeit/muq2/build/examples/SamplingAlgorithms/MCMC/MLDA/cpp
./MLDA "GLOBAL_CONFIG_PATH"
```
