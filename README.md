# oOo

oOo is a High-Performance Parallel Magnetic Tweezer program for real-time monitoring of protein folding and unfolding under force.

##### Usual Method [Sequential and single threaded]

![](https://raw.githubusercontent.com/Ojas-Singh/oOo/master/docs/1.PNG)



##### Our Method [Concurrent]

![](https://raw.githubusercontent.com/Ojas-Singh/oOo/master/docs/2.PNG)



# System and Hardware Requirements
+ Windows
+ Ximea camera (xiQ USB 3.0 SuperSpeed)
+ Piezo Controller (P-725.xDDPIFOC adnd PI-E-709)
  ( We tested on AMD threadripper Pro 3975wx 32 core 3.5Ghz 64GB RAM 64Bit Windows 10 Pro 20H2 )

## Installation

To get started with this project, you'll need to have Python 3.8 and its dependencies installed on your machine. Here's how you can do it:

1. **Python 3.8**: Visit the official Python website at [python.org](https://www.python.org) and download the Python 3.8 installer for your operating system.
2. **Ximea Drivers** : Visit [Ximea](https://www.ximea.com/support/wiki/apis/Python) and install the camera drivers then install the [xiAPI](https://www.ximea.com/support/wiki/apis/XIMEA_Linux_Software_Package#Installation).
 ( We have used https://www.ximea.com/support/attachments/37/XIMEA_API_Installer.exe with all API option ticked while installing and the used this guide to install ximea python API https://www.ximea.com/support/wiki/apis/Python_inst_win )
    ```bash
    python camtest.py
    ```
    if this return no error we good to go!
3. **Dependencies**: Once Python 3.8 is installed, open a terminal or command prompt and run the following command to install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    This command will install all the necessary dependencies listed in the `requirements.txt` file.

## Usage

To run the program, use the following command:

```bash
python oOo.py [arg1] [arg2]
```
```
Usage: [arg1] can be 
    fresh       ->     To take fresh stack.
    start       ->     Start data aquisition.
    plot        ->     Plot the collected data with [arg2] as bead number.
    stackplot   ->     Plot stack graph.
```

## Modification

You can modify specific settings in the config.py
the default settings are:
```
stack_size = 200        # Size of the stack
exposure = 1000         # Exposure time in milliseconds
resolution = 256        # Resolution of the image
workers = 20            # Number of workers for processing
driftworkers = 2        # Number of workers for drift correction
qu_limit = 100          # Queue limit for processing
driftworker = 4         # Number of workers for drift calculation

```


# Cite

Pritam Saha, Vishavdeep Vashisht, Ojas Singh, Amin Sagar, Gaurav Bhati, Surbhi Garg, Sabyasachi Rakshi. Exploring Force-Driven Stochastic Folding Dynamics in Mechano-Responsive Proteins and Implications in Phenotypic Variation. (In press)

Please feel free to contact the corresponding author (srakshit@iisermohali.ac.in) through mail in case of any difficulty in installation and running the codes.



