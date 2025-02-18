# Hurricane TRACK Forecast Uncertainty with Neural Networks
***
Neural networks are used to estimate hurricane track errors in the form of bivariate normal distributions.

## Tensorflow Code
This code was written in python 3.9.7, tensorflow 2.7.0, tensorflow-probability 0.15.0 and numpy 1.19.5.

## Order of Operations
1. run `python setup.py` to create the correct directory structure
2. define new experiments using `experiment_settings.py`
    * import and create an instance of the `Experiments` class:
    ```
    import experiment_settings
    exp_class = experiment_settings.Experiments()
    ```
    * define and save a new experiment (experiment dictionaries are saved to `experiments.json`):
    ```
    new_exp = exp_class.make_exp_dictionary(expname='exp_name', basin='AL')
    exp_class.add_experiment(new_exp)
    ```
    * all options and defaults are described in the `make_exp_dictionary` function in `experiment_settings.py` -- some likely settings to change include:
        * predictand (str): either "OFD" for official forecast error, or "OBD" for early forecast error
        * uncertainty (str): "centered_bivariate_normal" (do not fit location parameters) or "bivariate_normal" (do fit location parameters)
        * test_condition (str): "years" tests on the years in years_test (see next bullet), "leave-one-out" runs through all years, None does not test the network
        * years_test (list): which year(s) to use for testing (if above is "years", else ignored)
        * x_names (list): predictors to use -- predictor names that are already in default_x_names (defined in `Experiments` init) removes them, otherwise they are added
3. run `python _train_driver.py` to train networks
    * you will be presented with a list of the names of all saved experiments and asked to enter the experiment name you want to train -- this will train (and predict the testing set) for all basins and lead times associated with this experiment
    * you will then be asked if would like to overwrite the model and predictions -- this will be asked even if this is the first time training -- true to overwrite, anything else to not overwrite
    * saved model directories (one for each basin, lead time, and testing year) will be in the `saved_models` directory in a sub-directory with the experiment name
    * similarly, each saved prediction csv will be in the `saved_predictions` directory in a sub-directory with the experiment name


## Python Environment
The following python environment was used to implement this code. If you are running on a Linux machine, please also follow the `Compatibility Notes` provided in the following section to ensure you are using the latest version of `GLIBCXX`. 
```
conda create --name env-tcane python=3.9
conda activate env-tcane
pip install tensorflow==2.7.0
pip install tensorflow-probability==0.15.0
pip install scipy==1.9.1
pip install pandas==1.4.4
pip install matplotlib==3.5.3
pip install statsmodels==0.13.0
pip install --upgrade seaborn
pip install --upgrade palettable progressbar2 tabulate icecream flake8
pip install scikit-learn==1.6.1
pip install keras-tuner==1.4.7
pip install --upgrade jupyterlab black isort jupyterlab_code_formatter
pip install silence-tensorflow==1.2.3
pip install tqdm
pip install numpy==1.21.0
```
Note that `tensorflow` does not always play nicely with other packages. Common errors are with the `contourpy`, `protobuf`, and `numpy` libraries. The package versions we have provided here worked on our machine; if they don't work for you, you may need to upgrade or downgrade a version of a package to get it to work. And don't neglect the next section if you are on a Linux machine. 

## Compatibility Notes
For Linux users, the code requires libstdc++.so.6 version >= 3.4.29.  
1. To find the version installed on your system, run `strings /usr/lib64/libstdc++.so.6 | grep ^GLIBCXX_3.4.2`
2. To find the version installed in your Anaconda or Miniconda, run `strings /home/your-username/path-to-conda/anaconda3/lib/libstdc++.so.6 | grep ^GLIBCXX_3.4.2`
3. To use the latest version of GLIBCXX, add LD_LIBRARY path to your .bashrc. That can be done with the command: `echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/miniconda3/lib'>>~/.bashrc`

***

### Credits
This work is a collaborative effort between [Dr. Elizabeth A. Barnes](https://barnes.atmos.colostate.edu), [Dr. Randal J. Barnes](https://cse.umn.edu/cege/randal-j-barnes), [Dr. Martin A. Fernandez](https://mafern.github.io/), and Dr. Mark DeMaria, as well as other colleagues at CIRA.

### Funding sources

This work was supported by NOAA grant NA22OAR4590525.

### References
[1] Barnes, Elizabeth A., Randal J. Barnes and Nicolas Gordillo, 2021: Adding Uncertainty to Neural Network Regression Tasks in the Geosciences, arXiv 2109.07250.

[2] Barnes, Elizabeth A., Randal J. Barnes and Mark DeMaria, 2022: Sinh-arcsinh-normal distributions to add uncertainty to neural network regression tasks: applications to tropical cyclone intensity forecasts, preprint available at https://doi.org/10.31223/X51649.

### License
This project is licensed under an MIT license.

MIT © [Elizabeth A. Barnes](https://github.com/eabarnes1010)
