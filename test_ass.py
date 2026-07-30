import uuid
from pathlib import Path

def _format_ass_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    if cs == 100:
        s += 1
        cs = 0
        if s == 60:
            s = 0
            m += 1
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

print(_format_ass_time(1.234))
print(_format_ass_time(61.999))
