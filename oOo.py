import sys,os
from termcolor import colored
def help():
    print("Usage:")
    print("python oOo.py [arg1] [arg2] ...")
    print("options for [arg1] :")
    print("             fresh       ->     To take fresh stack.")
    print("             start       ->     start data aquisition.")
    print("             plot        ->     plot the collected data. with [arg2] as bead number.")
    print("             stackplot   ->     Plot stack graph.")
if __name__ == '__main__':

    print(colored(" ", 'yellow')    )                                                
    print(colored("  ", 'yellow')     )                                              
    print(colored("                      OOOOOOOOO                      ", 'yellow'))
    print(colored("                    OO:::::::::OO                    ", 'yellow'))
    print(colored("                  OO:::::::::::::OO                  ", 'yellow'))
    print(colored("                 O:::::::OOO:::::::O                 ", 'yellow'))
    print(colored("   ooooooooooo   O::::::O   O::::::O   ooooooooooo   ", 'yellow'))
    print(colored(" oo:::::::::::oo O:::::O     O:::::O oo:::::::::::oo ", 'yellow'))
    print(colored("o:::::::::::::::oO:::::O     O:::::Oo:::::::::::::::o", 'yellow'))
    print(colored("o:::::ooooo:::::oO:::::O     O:::::Oo:::::ooooo:::::o", 'yellow'))
    print(colored("o::::o     o::::oO:::::O     O:::::Oo::::o     o::::o", 'yellow'))
    print(colored("o::::o     o::::oO:::::O     O:::::Oo::::o     o::::o", 'yellow'))
    print(colored("o::::o     o::::oO:::::O     O:::::Oo::::o     o::::o", 'yellow'))
    print(colored("o::::o     o::::oO::::::O   O::::::Oo::::o     o::::o", 'yellow'))
    print(colored("o:::::ooooo:::::oO:::::::OOO:::::::Oo:::::ooooo:::::o", 'yellow'))
    print(colored("o:::::::::::::::o OO:::::::::::::OO o:::::::::::::::o", 'yellow'))
    print(colored(" oo:::::::::::oo    OO:::::::::OO    oo:::::::::::oo ", 'yellow'))
    print(colored("   ooooooooooo        OOOOOOOOO        ooooooooooo   ", 'yellow'))
    print(colored("                                                     ", 'yellow'))    
    print(colored(" ", 'yellow') )

    filename = ""

    if len(sys.argv) > 1:
        if sys.argv[1] == "fresh":
            filename = "live-view.py"
        if sys.argv[1] == "start":
            filename = "main.py"
        if sys.argv[1] == "plot":
            filename = "clean.py"+" "+str(sys.argv[2])
        if sys.argv[1] == "stackplot":
            filename = "stackplot.py"
        print("Opening ",filename)
        os.system("python.exe "+filename)     
                
    else:
        help()
    