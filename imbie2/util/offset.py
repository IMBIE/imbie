def apply_offset(t, mass, mid):
    i_mid = 0
    near = mid
    for i, it in enumerate(t):
        diff = abs(it - mid)
        if diff < near:
            i_mid = i
            near = diff

    offset = mass[i_mid]
    return mass - offset
