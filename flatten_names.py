#!/usr/bin/env python3

# https://askubuntu.com/questions/746860/rename-a-file-to-parent-directorys-name-in-terminal

import shutil
import os
import sys

dr = sys.argv[1]
filename = sys.argv[2]
append2 = sys.argv[3]

for root, dirs, files in os.walk(dr):
    for file in files:
        if file == filename:
            filetype = file.split(".")[-1]

            spl = root.split("/"); newname = spl[-1]; sup = ("/").join(spl[:-1])
            shutil.move(root+"/"+file, sup+"/"+newname+f"_{append2}.{filetype}"); shutil.rmtree(root)
