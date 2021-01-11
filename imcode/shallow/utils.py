def state_to_index(state):
    pass
def index_to_state(t,index,folded=False):
    if folded:
        pass
    else:
        return bin(index+2**(2*t))[3:].replace("1","u").replace("0","d")

def unfold_state(state):
    acf="".join(["u" if c=="b" or c=="+" else "d" for c in state])
    acb="".join(["u" if c=="a" or c=="+" else "d" for c in state])
    return acf,acb[::-1]
def fold_state(fw,bw):
    dicti={("u","u"):"+",("u","d"):"b",("d","u"):"a",("d","d"):"-"}
    return "".join([dicti[cs] for cs in zip(fw,bw[::-1])])
