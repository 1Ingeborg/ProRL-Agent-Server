#!/usr/bin/env python3
import re, sys, time
src, dst = sys.argv[1:3]
ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
with open(src, "rb") as inp, open(dst, "w", encoding="utf-8") as out:
    while True:
        chunk = inp.readline()
        if not chunk:
            time.sleep(0.25)
            continue
        line = chunk.decode("utf-8", "replace").replace("\r", "\n")
        out.write(ansi.sub("", line))
        out.flush()
