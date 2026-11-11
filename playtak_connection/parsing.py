import re

FILES = "abcdefgh"

# PTN placement: optional stone letter (F/S/C) + square, e.g. "a1", "Sb3", "Cc2"
_PLACE_RE = re.compile(r"([FSC]?)([a-h][1-8])")
# PTN stack move: optional count, start square, direction, optional drops, e.g. "3c3>12", "c3+"
_MOVE_RE = re.compile(r"([1-8]?)([a-h])([1-8])([-+<>])([1-8]*)")


def ptn_to_server(ptn: str) -> str:
    """Convert a PTN move to Playtak wire format (without the 'Game#N ' prefix).

    'a1'     -> 'P A1'
    'Sb3'    -> 'P B3 W'
    'Cc2'    -> 'P C2 C'
    '3c3>12' -> 'M C3 E3 1 2'
    'c3+'    -> 'M C3 C4 1'
    """
    ptn = ptn.strip().rstrip("*?!'")  # drop tak/annotation marks like "*" or "?!"

    if m := _PLACE_RE.fullmatch(ptn):
        stone, square = m.groups()
        out = f"P {square.upper()}"
        if stone == "S":
            out += " W"      # Playtak calls standing stones "walls"
        elif stone == "C":
            out += " C"
        return out           # flats ('' or 'F') have no suffix

    if m := _MOVE_RE.fullmatch(ptn):
        count, file, row, direction, drops = m.groups()
        count = count or "1"                  # "c3+" lifts one stone
        drop_list = list(drops) or [count]    # "3c3>" drops everything on the next square
        if sum(map(int, drop_list)) != int(count):
            raise ValueError(f"drops don't add up to the lifted count: {ptn!r}")

        n = len(drop_list)                    # number of squares travelled
        r, f = int(row), FILES.index(file)
        if direction == "+":
            r += n
        elif direction == "-":
            r -= n
        elif direction == ">":
            f += n
        else:  # "<"
            f -= n
        if not (0 <= f < 8 and 1 <= r <= 8):
            raise ValueError(f"move leaves the board: {ptn!r}")

        start = f"{file.upper()}{row}"
        end = f"{FILES[f].upper()}{r}"
        return f"M {start} {end} {' '.join(drop_list)}"

    raise ValueError(f"unrecognised PTN move: {ptn!r}")


def server_to_ptn(cmd: str, explicit_flat: bool = False, verbose: bool = False) -> str:
    """Convert a Playtak wire move to PTN.

    'P A1'        -> 'a1'   (or 'Fa1' if explicit_flat=True)
    'P B3 W'      -> 'Sb3'
    'P C2 C'      -> 'Cc2'
    'M C3 E3 1 2' -> '3c3>12'
    'M C3 C4 1'   -> 'c3+'  (or '1c3+1' if verbose=True)
    """
    w = cmd.strip().split()

    if w[0] == "P":
        square = w[1].lower()
        kind = w[2] if len(w) > 2 else ""
        if kind == "W":
            return "S" + square
        if kind == "C":
            return "C" + square
        return ("F" if explicit_flat else "") + square

    if w[0] == "M":
        start, end, drops = w[1], w[2], w[3:]
        sf, sr = start[0].upper(), int(start[1])
        ef, er = end[0].upper(), int(end[1])

        if start.upper() == end.upper():
            raise ValueError(f"start and end are the same square: {cmd!r}")
        if sf != ef and sr != er:
            raise ValueError(f"diagonal move: {cmd!r}")

        if sr != er:
            direction = "+" if er > sr else "-"
        else:
            direction = ">" if ef > sf else "<"

        lifted = sum(int(d) for d in drops)
        count = "" if (lifted == 1 and not verbose) else str(lifted)
        drop_str = "" if (len(drops) == 1 and not verbose) else "".join(drops)
        return f"{count}{start.lower()}{direction}{drop_str}"

    raise ValueError(f"unknown Playtak move: {cmd!r}")


if __name__ == "__main__":
    for ptn in ["a1", "Sb3", "Cc2", "3c3>12", "c3+", "3d4-21", "5e5<"]:
        wire = ptn_to_server(ptn)
        back = server_to_ptn(wire)
        print(f"{ptn:8} -> {wire:14} -> {back}")
        assert back == ptn, (ptn, back)