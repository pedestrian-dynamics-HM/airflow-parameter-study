import os.path

# Directories
cwd = os.path.dirname(os.path.realpath(__file__))
path2root = os.path.join(cwd, os.pardir, os.pardir, os.pardir)

# set new working directory
os.chdir(path2root)
path2root = os.getcwd()
root2scr = "vimuq"

root2experiments = os.path.join(root2scr, "experiments")
root2output = os.path.join(root2experiments, "output")
