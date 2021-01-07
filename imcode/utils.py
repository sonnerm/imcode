def popcount(i):
    count = 0
    while i:
        i &= i - 1
        count += 1
    return count
