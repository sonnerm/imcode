def pattern_to_mps(pattern):
    pdict={"+":[1,0,0,0],"-":[0,0,0,1],"b":[0,1,0,0],"a":[0,0,1,0],"q":[0,1,1,0],"c":[1,0,0,1],"*":[1,1,1,1]}
    state = [pdict[p] for p in pattern]
    psi = MPS.from_product_state(state)
    return psi
