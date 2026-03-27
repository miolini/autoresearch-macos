"""
Live training monitor. Run in a separate terminal:
    uv run monitor.py
"""
import os, re, time, sys, subprocess, ctypes, ctypes.util

CACHE_DIR = os.environ.get("AUTORESEARCH_CACHE", os.path.expanduser("~/.cache/autoresearch-10k"))
LOG_FILE = os.path.join(os.path.dirname(__file__), "run.log")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.tsv")

# ANSI colors
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
MAGENTA = "\033[35m"
WHITE = "\033[37m"
BG_GREEN = "\033[42m"
BG_GRAY = "\033[100m"
BG_CYAN = "\033[46m"
BG_YELLOW = "\033[43m"
BG_RED = "\033[41m"
BG_MAGENTA = "\033[45m"

# === Key benchmarks ===
BENCHMARKS = {
    'climbmix_general':  {'val_bpb': 2.146, 'label': 'ClimbMix (general text)',     'color': DIM},
    '10k_first_run':     {'val_bpb': 1.711, 'label': '10-K Baseline (fan-cooled)',  'color': YELLOW},
    '10k_best':          {'val_bpb': None,   'label': '10-K Best (current)',         'color': GREEN},
}

def clear_screen():
    print("\033[2J\033[H", end="")

def get_thermal_info():
    """Get battery temp and macOS thermal state (no sudo needed)."""
    info = {'temp_c': None, 'state': None}
    try:
        r = subprocess.run(['ioreg', '-rc', 'AppleSmartBattery'],
                           capture_output=True, text=True, timeout=3)
        for line in r.stdout.split('\n'):
            if '"Temperature"' in line and 'Shutdown' not in line and 'Virtual' not in line:
                info['temp_c'] = int(line.split('=')[1].strip()) / 100
                break
    except Exception:
        pass
    try:
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library('objc'))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.objc_msgSend.restype = ctypes.c_void_p
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        NSProcessInfo = objc.objc_getClass(b'NSProcessInfo')
        processInfo = objc.objc_msgSend(NSProcessInfo, objc.sel_registerName(b'processInfo'))
        objc.objc_msgSend.restype = ctypes.c_long
        state = objc.objc_msgSend(processInfo, objc.sel_registerName(b'thermalState'))
        info['state'] = {0: 'Nominal', 1: 'Fair', 2: 'Serious', 3: 'Critical'}.get(state, f'Unknown({state})')
    except Exception:
        pass
    return info

def draw_temp_meter(temp_c, state, width=40):
    """Draw a temperature gauge with danger zones."""
    if temp_c is None:
        return f"  {DIM}Temperature: unavailable{RESET}"

    t_min, t_max = 20, 70
    pct = max(0, min(1, (temp_c - t_min) / (t_max - t_min)))
    filled = int(width * pct)

    if temp_c < 35:
        color, zone = CYAN, "COOL"
    elif temp_c < 45:
        color, zone = GREEN, "WARM"
    elif temp_c < 55:
        color, zone = YELLOW, "HOT"
    else:
        color, zone = RED, "DANGER"

    bar = ""
    for i in range(width):
        pos_temp = t_min + (i / width) * (t_max - t_min)
        if i < filled:
            if pos_temp < 35:    bar += f"{CYAN}\u2588{RESET}"
            elif pos_temp < 45:  bar += f"{GREEN}\u2588{RESET}"
            elif pos_temp < 55:  bar += f"{YELLOW}\u2588{RESET}"
            else:                bar += f"{RED}\u2588{RESET}"
        else:
            bar += f"{DIM}\u2591{RESET}"

    state_color = GREEN if state == 'Nominal' else YELLOW if state == 'Fair' else RED

    lines = []
    lines.append(f"  {DIM}20C{RESET} {bar} {DIM}70C{RESET}")
    lines.append(f"  {color}{BOLD}{temp_c:.1f}C{RESET} ({zone}) | macOS: {state_color}{state}{RESET}")

    if state in ('Serious', 'Critical'):
        lines.append(f"  {RED}{BOLD}!! THERMAL THROTTLING ACTIVE !!{RESET}")
    elif state == 'Fair' or temp_c > 45:
        lines.append(f"  {YELLOW}Approaching thermal limits{RESET}")

    return "\n".join(lines)

def parse_log_line(line):
    m = re.search(
        r'step (\d+) \(([\d.]+)%\) \| loss: ([\d.]+) \| lrm: ([\d.]+) \| dt: (\d+)ms \| tok/sec: ([\d,]+).*?remaining: (\d+)s',
        line
    )
    if m:
        return {
            'step': int(m.group(1)),
            'pct': float(m.group(2)),
            'loss': float(m.group(3)),
            'lrm': float(m.group(4)),
            'dt_ms': int(m.group(5)),
            'tok_sec': int(m.group(6).replace(',', '')),
            'remaining': int(m.group(7)),
        }
    return None

def parse_final(lines):
    results = {}
    for line in lines:
        for key in ['val_bpb', 'num_steps', 'total_tokens_M', 'training_seconds']:
            if line.startswith(f'{key}:'):
                results[key] = line.split(':')[1].strip()
    return results

def get_best_bpb():
    """Scan git log for the best val_bpb we've achieved."""
    try:
        r = subprocess.run(['git', 'log', '--oneline', '--all'],
                           capture_output=True, text=True, timeout=5,
                           cwd=os.path.dirname(__file__))
        best = None
        for line in r.stdout.split('\n'):
            m = re.search(r'val_bpb=([\d.]+)', line)
            if m:
                val = float(m.group(1))
                if best is None or val < best:
                    best = val
        return best
    except Exception:
        return None

def draw_progress_bar(pct, width=50):
    filled = int(width * pct / 100)
    bar = f"{BG_GREEN} {RESET}" * filled + f"{BG_GRAY} {RESET}" * (width - filled)
    return f"{bar} {pct:5.1f}%"

def draw_loss_chart(losses, width=55, height=10):
    if len(losses) < 2:
        return "  Waiting for data..."

    if len(losses) > width:
        step = len(losses) / width
        losses = [losses[int(i * step)] for i in range(width)]

    mn, mx = min(losses), max(losses)
    if mx == mn:
        mx = mn + 1

    lines = []
    for row in range(height):
        threshold = mx - (row / (height - 1)) * (mx - mn)
        line = ""
        for val in losses:
            if val >= threshold:
                line += f"{GREEN}\u2588{RESET}"
            else:
                line += " "

        if row == 0:       label = f" {mx:.2f}"
        elif row == height - 1: label = f" {mn:.2f}"
        else:              label = ""
        lines.append(f"  {DIM}\u2502{RESET}{line}{DIM}{label}{RESET}")

    lines.append(f"  {DIM}\u2514{'─' * len(losses)}{RESET}")
    return "\n".join(lines)

def draw_benchmark_race(current_bpb, best_bpb, finished=False):
    """Draw a visual race track showing where we are vs benchmarks."""
    climbmix = 2.146
    baseline = 1.711
    best = best_bpb or baseline

    # The "finish line" — theoretical perfect compression
    target = 1.0
    worst = climbmix + 0.1  # pad slightly

    lines = []
    lines.append(f"  {BOLD}Benchmark Race{RESET} {DIM}(lower val_bpb = better){RESET}")
    lines.append("")

    width = 50

    def pos(val):
        return int(width * (1 - (val - target) / (worst - target)))

    def draw_marker_line(val, label, color, marker="*"):
        p = max(0, min(width - 1, pos(val)))
        line = [f"{DIM}.{RESET}"] * width
        line[p] = f"{color}{BOLD}{marker}{RESET}"
        bar = "".join(line)
        improvement = (1 - val / climbmix) * 100
        return f"  {bar} {color}{val:.3f}{RESET} {DIM}{label} ({improvement:+.1f}%){RESET}"

    # Draw each benchmark
    lines.append(draw_marker_line(climbmix, "General (ClimbMix)", RED, "\u2716"))
    lines.append(draw_marker_line(baseline, "10-K Baseline", YELLOW, "\u25cf"))

    if best < baseline:
        lines.append(draw_marker_line(best, "10-K Best", GREEN, "\u2605"))

    if current_bpb and not finished:
        lines.append(draw_marker_line(current_bpb, "Current run (est.)", CYAN, "\u25b6"))

    # Scale labels
    arrow = "\u2192"
    target_pad = pos(target) + 6
    worse_pad = width - pos(target)
    lines.append(f"  {DIM}{'target':>{target_pad}}{'worse ' + arrow:>{worse_pad}}{RESET}")

    # Summary
    lines.append("")
    improvement_from_general = (1 - best / climbmix) * 100
    improvement_from_baseline = (1 - best / baseline) * 100

    if best < baseline:
        lines.append(f"  {GREEN}{BOLD}\u2714 Beating baseline by {improvement_from_baseline:.1f}%{RESET}")
    else:
        lines.append(f"  {YELLOW}\u2192 At baseline level{RESET}")

    lines.append(f"  {BOLD}\u2193 {abs(improvement_from_general):.1f}% better{RESET} than general-purpose model at financial text")

    return "\n".join(lines)

def draw_journey(best_bpb):
    """Show the improvement journey as a simple visual."""
    climbmix = 2.146
    baseline = 1.711
    best = best_bpb or baseline

    total_possible = climbmix - 1.0  # from general model to "perfect"
    achieved = climbmix - best
    pct = achieved / total_possible * 100

    lines = []
    lines.append(f"  {BOLD}Optimization Journey{RESET}")

    # Milestones
    milestones = [
        (2.146, "General model",     DIM,    True),
        (1.711, "Specialized",       YELLOW, True),
        (1.500, "Well-tuned",        GREEN,  best <= 1.500),
        (1.300, "Highly optimized",  CYAN,   best <= 1.300),
        (1.000, "Near-perfect",      MAGENTA, best <= 1.000),
    ]

    for val, label, color, reached in milestones:
        if reached:
            marker = f"{color}{BOLD}\u2714{RESET}"
        elif best < val + 0.15:
            marker = f"{YELLOW}\u25b6{RESET}"  # approaching
        else:
            marker = f"{DIM}\u25cb{RESET}"

        current = ""
        if reached and (best >= val or val == milestones[-1][0]):
            # Find if this is the current milestone
            pass

        lines.append(f"  {marker} {color}{val:.3f}{RESET}  {label}")

    # Current position indicator
    lines.append(f"  {GREEN}{BOLD}\u2192 You are here: {best:.3f}{RESET}")

    return "\n".join(lines)

def main():
    while True:
        try:
            clear_screen()

            # Header
            print(f"{BOLD}{CYAN}{'='*70}{RESET}")
            print(f"{BOLD}{CYAN}  AUTORESEARCH 10-K FINANCIAL SLM TRAINING MONITOR{RESET}")
            print(f"{BOLD}{CYAN}{'='*70}{RESET}\n")

            # Get best historical result
            best_bpb = get_best_bpb()

            # Read log
            steps_data = []
            all_lines = []
            finished = False

            if os.path.exists(LOG_FILE):
                with open(LOG_FILE) as f:
                    content = f.read()
                all_lines = content.strip().split('\n') if content.strip() else []

                for line in all_lines:
                    parsed = parse_log_line(line)
                    if parsed:
                        steps_data.append(parsed)

                if any('val_bpb' in line for line in all_lines):
                    finished = True

            # ── Benchmark Race ──
            current_bpb = None
            if finished:
                finals = parse_final(all_lines)
                try:
                    current_bpb = float(finals.get('val_bpb', 0))
                    if best_bpb is None or current_bpb < best_bpb:
                        best_bpb = current_bpb
                except ValueError:
                    pass

            print(draw_benchmark_race(current_bpb, best_bpb, finished))
            print()

            # ── Current Run Status ──
            print(f"  {BOLD}{'─'*66}{RESET}")

            if not steps_data and not finished:
                print(f"\n  {YELLOW}Waiting for training to start...{RESET}")
                print(f"  {DIM}(watching {LOG_FILE}){RESET}")
                time.sleep(2)
                continue

            if finished:
                finals = parse_final(all_lines)
                val_bpb = finals.get('val_bpb', '?')
                val_float = float(val_bpb) if val_bpb != '?' else None

                # Did this run improve?
                if val_float and best_bpb and val_float <= best_bpb:
                    verdict = f"{GREEN}{BOLD}\u2714 NEW BEST!{RESET}"
                elif val_float and best_bpb and val_float < 1.711:
                    verdict = f"{YELLOW}\u2192 Better than baseline, not best{RESET}"
                else:
                    verdict = f"{RED}\u2716 No improvement{RESET}"

                print(f"\n  {BOLD}Status:{RESET}  {GREEN}COMPLETED{RESET}  {verdict}")
                print(f"  {BOLD}val_bpb:{RESET} {GREEN}{BOLD}{val_bpb}{RESET}  |  Steps: {finals.get('num_steps', '?')}  |  Tokens: {finals.get('total_tokens_M', '?')}M")
                print()

            elif steps_data:
                latest = steps_data[-1]
                mins_left = latest['remaining'] // 60
                secs_left = latest['remaining'] % 60

                print(f"\n  {BOLD}Status:{RESET}  {YELLOW}TRAINING{RESET}  step {latest['step']}  |  loss {latest['loss']:.4f}  |  LR x{latest['lrm']:.2f}")
                print(f"  {BOLD}Speed:{RESET}   {latest['tok_sec']:,} tok/sec  ({latest['dt_ms']}ms/step)  |  ETA: {mins_left}m {secs_left}s")
                print()
                print(f"  {draw_progress_bar(latest['pct'])}")
                print()

            # ── Loss Chart ──
            if steps_data:
                losses = [s['loss'] for s in steps_data]
                print(f"  {BOLD}Loss Curve:{RESET}")
                print(draw_loss_chart(losses))
                print()

            # ── Thermal ──
            thermal = get_thermal_info()
            print(f"  {BOLD}Thermal:{RESET}")
            print(draw_temp_meter(thermal['temp_c'], thermal['state']))
            print()

            # ── Journey / Milestones ──
            if best_bpb:
                print(draw_journey(best_bpb))
                print()

            # Footer
            ts = time.strftime("%H:%M:%S")
            print(f"  {DIM}Last updated: {ts} | Refreshing every 3s | Ctrl+C to exit{RESET}")

            time.sleep(3)

        except KeyboardInterrupt:
            print(f"\n{DIM}Monitor stopped.{RESET}")
            break
        except Exception as e:
            print(f"  {RED}Error: {e}{RESET}")
            import traceback
            traceback.print_exc()
            time.sleep(3)

if __name__ == "__main__":
    main()
