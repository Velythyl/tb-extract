#!/usr/bin/env python3

# https://askubuntu.com/questions/746860/rename-a-file-to-parent-directorys-name-in-terminal

import shutil
import os
import sys

dr = sys.argv[1]

for root, dirs, files in os.walk(dr):
    for file in files:
        if file == "Maze_0.95.png":
            filetype = file.split(".")[-1]

            spl = root.split("/"); newname = spl[-1]; sup = ("/").join(spl[:-1])
            shutil.move(root+"/"+file, sup+"/"+newname+f".{filetype}"); shutil.rmtree(root)
