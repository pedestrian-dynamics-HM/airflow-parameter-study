import math


def round_up_to_xth_place(number, place=None):
    """
    Example:
    for place=None: rounds 78 to 80, 178 to 200, 8178 to 9000
    for place=1:    rounds 78 to 80, 178 to 180, 8178 to 8180
    for place=2:    rounds 78 to 100, ...
    """
    if place is None:
        place = 10 ** math.floor(math.log10(number))
    else:
        place = 10**place

    return math.ceil((number / place)) * place
