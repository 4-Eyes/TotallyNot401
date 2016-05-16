import os, re, getpass


def disable_folders(dir, regex_string=None):
    if regex_string:
        r = re.compile(regex_string)
    for d in [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]:
        if ((regex_string and r.search(d)) or not regex_string) and "_disabled" not in d:
            os.rename(os.path.join(dir, d), os.path.join(dir, d + "_disabled"))


def enable_folders(dir, regex_string=None):
    if regex_string:
        r = re.compile(regex_string)
    for d in [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]:
        if ((regex_string and r.search(d)) or not regex_string) and "_disabled" in d:
            os.rename(os.path.join(dir, d), os.path.join(dir, d.replace("_disabled", "")))

if getpass.getuser() == "Matthew":
    disable_folders("./test", "(other)|(rat)|(possum)")
    disable_folders("./valid", "(other)|(rat)|(possum)")
    disable_folders("./train", "(other)|(rat)|(possum)")
    #enable_folders("./cacophony-of-problems/valid")
    #enable_folders("./cacophony-of-problems/test")
    #enable_folders("./cacophony-of-problems/train")
else:
    # disable_folders("/media/james/9a6d3124-40f4-4227-9ef6-5cecdc794447/Reference Images/test", "(other)|(rat)|(possum)")
    # disable_folders("/media/james/9a6d3124-40f4-4227-9ef6-5cecdc794447/Reference Images/valid", "(other)|(rat)|(possum)")
    # disable_folders("/media/james/9a6d3124-40f4-4227-9ef6-5cecdc794447/Reference Images/train", "(other)|(rat)|(possum)")
    enable_folders("/media/james/9a6d3124-40f4-4227-9ef6-5cecdc794447/Reference Images/valid")
    enable_folders("/media/james/9a6d3124-40f4-4227-9ef6-5cecdc794447/Reference Images/test")
    enable_folders("/media/james/9a6d3124-40f4-4227-9ef6-5cecdc794447/Reference Images/train")
