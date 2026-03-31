"""
AETHER v3 — Startup Banner
Avatar-style logo with ANSI color codes.
"""


def print_banner():
    R = '\033[91m'   # red
    W = '\033[97m'   # white
    D = '\033[90m'   # dark grey / black
    B = '\033[1m'    # bold
    X = '\033[0m'    # reset

    print(f"""
{D}    ████████████████████████████████{X}
{D}   █{R}██████████████████████████████{D}█{X}
{D}  █{R}████{W}╔═══════════════════╗{R}████{D}█{X}
{D}  █{R}████{W}║  {B}{R}▄████▄   ▄████▄{W}  ║{R}████{D}█{X}
{D}  █{R}████{W}║ {B}{R}████████████████{W} ║{R}████{D}█{X}
{D}  █{R}████{W}║ {B}{W}██{R}██████████{W}██{W} ║{R}████{D}█{X}
{D}  █{R}████{W}║ {B}{W}█{D}▀▀{W}████████{D}▀▀{W}█{W} ║{R}████{D}█{X}
{D}  █{R}████{W}║  {B}{R}▀████▀   ▀████▀{W}  ║{R}████{D}█{X}
{D}  █{R}████{W}║{D}▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄{W}║{R}████{D}█{X}
{D}  █{R}████{W}║{B}{W}  ━━━━━━━━━━━━━━━  {W}║{R}████{D}█{X}
{D}  █{R}████{W}║{B}{R}  ▌ AETHER  v3 ▐  {W}║{R}████{D}█{X}
{D}  █{R}████{W}║{B}{W}  ━━━━━━━━━━━━━━━  {W}║{R}████{D}█{X}
{D}  █{R}████{W}╚═══════════════════╝{R}████{D}█{X}
{D}   █{R}██████████████████████████████{D}█{X}
{D}    ████████████████████████████████{X}

{D}  ┌─────────────────────────────────────────┐{X}
{D}  │{W}  Adaptive Embodied Task Hierarchy       {D}│{X}
{D}  │{W}  for Executable Robotics               {D}│{X}
{D}  │{R}  DRL-First Hybrid FDIR · v3.0          {D}│{X}
{D}  │{W}  Multi-Agent · Self-Correcting          {D}│{X}
{D}  └─────────────────────────────────────────┘{X}
""")

    # Animated initialization line
    import time
    import sys
    sys.stdout.write(f'{D}  Initializing autonomous systems {X}')
    sys.stdout.flush()
    for i in range(3):
        for frame in [f'{R}●{D}○○', f'{R}●●{D}○', f'{R}●●●']:
            sys.stdout.write(f'\r{D}  Initializing autonomous systems {frame}{X}')
            sys.stdout.flush()
            time.sleep(0.15)
    sys.stdout.write(f'\r{D}  Initializing autonomous systems {R}●●●{W} READY{X}\n\n')
    sys.stdout.flush()
