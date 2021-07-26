from correlator import correlator
#check equal-time Majorana self-correlations (correlations < (c + c^dagger)(c + c^dagger) > and <- (c - c^dagger)(c - c^dagger) > at equal times yield partition sum since (c + c^dagger)(c + c^dagger) = -(c - c^dagger)(c - c^dagger) = 1)

def test_identity_correlations(A, n_expect, ntimes):
    status = 0
    print ('Testing identity correlations..')
    for tau in range (ntimes):
        #first Majorana type (0, Theta, -)
        #forward branch
        test = correlator(A,n_expect, 0, 0, tau, 0, 0, tau)
        if (abs(test) - 1) > 1e-6:
            print('Identity correlation test not passed, forward branch, Majorana type 0 (Theta/- Majorana)',' tau=', tau, 'value: ', test) 
            status += 1

        #backward brnach
        test = correlator(A,n_expect, 1, 0, tau, 1, 0, tau)
        if (abs(test) - 1) > 1e-6:
            print('Identity correlation test not passed, backward branch, Majorana type 0 (Theta/- Majorana)',' tau=', tau, 'value: ', test) 
            status += 1


        #second Majorana type (1, Zeta, +)
        #forward branch
        test = correlator(A,n_expect, 0, 1, tau, 0, 1, tau)
        if (abs(test) - 1) > 1e-6:
            print('Identity correlation test not passed, forward branch, Majorana type 1 (Zeta/+ Majorana)',' tau=', tau, 'value: ', test) 
            status += 1
        #backward branch
        test = correlator(A,n_expect, 1, 1, tau, 1, 1, tau)
        if (abs(test) - 1) > 1e-6:
            print('Identity correlation test not passed, backward branch, Majorana type 1 (Zeta/+ Majorana)',' tau=', tau, 'value: ', test) 
            status += 1
    if status == 0:
        print ('Testing identity correlations sucessfully terminated..')    
    else: 
        print ('Identity correlation tests not passed in ',status, 'cases.' )


def anti_sym_check(matrix):
    dim = len(matrix[0])
    check = 0
    for i in range (dim):
        for j in range (i,dim):
            check += abs(matrix[i,j] + matrix[j,i])
    
    if check > 1e-6:
        print ('Antisymmetry test not passed..', check)
    
    else:
        print ('Antisymmetry test successfully passed..')