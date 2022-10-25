## Setup

1. Install requirements via 
```
pip install -r requirements.txt
```

2. Set the desired folder for logs/results at `confs/paths.yml`

3. The script that runs experiments can be found at `src/poc/run_expts.py`. The training code can be found at `src/poc/run_agent.py`. The parameters for the experiments in the paper are recorded here. By default, the script submits experiments to a Condor cluster. The script may be easily modifed to instead run experiments locally. 

4. The code to plot results can be found in a jupyter notebook inside the `notebooks` folder.