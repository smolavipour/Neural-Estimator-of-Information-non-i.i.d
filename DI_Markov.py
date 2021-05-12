import time
import numpy as np
import pickle
import argparse
import CMINE_lib as CMINE


################## Parsing simulation arguments ##################

parser = argparse.ArgumentParser(description='provide arguments for DI estimation')
parser.add_argument('--link', type=str, default=None, help='Example X_Y for I(X->Y), or X_Y-Z for I(X->Y||Z)')
args = parser.parse_args()

Link = args.link
n = int(2e5)

sigma_x = 3
sigma_y = 2
sigma_z = 1
a_1 = 1
a_2 = 2
A = np.asarray([[0,0,0],[0,0,0],[0,0,0]])
B = np.asarray([[0,0,0],[a_1,0,0],[0,a_2,0]])

params = (A,B,sigma_x,sigma_y,sigma_z)

if Link == 'X_Y':
    #Estimate I(X->Y) or equivalently I(X_1 X_2 X_3;Y_3|Y_1 Y_2)
    True_DI = 0.5*np.log(1+ a_1**2 * sigma_x**2/sigma_y**2)
    arrng = [[0,3,6],[1],[4,7]]
elif Link == 'X_Z':
    True_DI = 0.5*np.log(1+ (a_1**2 * a_2**2 * sigma_x**2 )/(a_2**2 * sigma_y**2 + sigma_z**2))
    arrng = [[0,3,6],[2],[5,8]]
elif Link == 'Y_X':
    True_DI = 0
    arrng = arrng = [[1,4,7],[0],[3,6]]
elif Link == 'Y_Z':
    True_DI = 0.5*np.log(1+ (a_1**2 * a_2**2 * sigma_x**2 + a_2**2 * sigma_y**2)/sigma_z**2)
    arrng = [[1,4,7],[2],[5,8]]
elif Link == 'Z_X':
    True_DI = 0   
    arrng = [[2,5,8],[0],[3,6]]
elif Link == 'Z_Y':
    True_DI = 0
    arrng = [[2,5,8],[1],[4,7]]
###########################################
elif Link == 'X_Y-Z':
    #Estimate I(X->Y||Z) or equivalently I(X_1 X_2 X_3;Y_3|Y_1 Y_2 Z_1 Z_2 Z_3)
    True_DI = 0.5*np.log(1+ a_1**2 * sigma_x**2/sigma_y**2)
    arrng = [[0,3,6],[1],[4,7,2,5,8]]
elif Link == 'X_Z-Y':
    True_DI = 0
    arrng = [[0,3,6],[2],[5,8,1,4,7]]
elif Link == 'Y_X-Z':
    True_DI = 0
    arrng = [[1,4,7],[0],[3,6,2,5,8]]
elif Link == 'Y_Z-X':
    True_DI = 0.5*np.log(1+ (a_2**2 * sigma_y**2)/sigma_z**2)
    arrng = [[1,4,7],[2],[5,8,0,3,6]]
elif Link == 'Z_X-Y':
    True_DI = 0   
    arrng = [[2,5,8],[0],[3,6,1,4,7]]
elif Link == 'Z_Y-X':
    True_DI = 0
    arrng = [[2,5,8],[1],[4,7,0,3,6]]
else:
    print('Wrong link input')
    pass


K = 2
b_size = n//2

# Set up neural network paramters
LR = 1e-3
EPOCH = 200
SEED = 123
input_size = 6
hidden_size = 64
num_classes = 2
tau = 1e-2

NN_params = (input_size,hidden_size,num_classes,tau)
EVAL = False

#Monte Carlo param
T = 20
S = 10


DI_DV = []

for s in range(S):
    DI_DV_t = []
    print('s=',s)    
    
    #Create dataset
    dataset = CMINE.create_dataset(GenModel='Markov_Gaussian_2', Params=params, N=n)
    dataset = CMINE.prepare_dataset(dataset, Mode='norm',S=4)
    #[x,y,z,x_,y_,z_,x__,y__,z__]

    for t in range(T): 
        print('t=',t)
        start_time = time.time()
        
        batch_train, target_train, joint_test, prod_test = CMINE.batch_construction(data=dataset, arrange=arrng, set_size=b_size, K_neighbor=K)    
        
        DI_DV_Eval=[]

        start_time = time.time()
        #Train
        if EVAL:
            model, loss_e, DI_LDR_e, DI_DV_e, DI_NWJ_e = CMINE.train_classifier(BatchTrain=batch_train, TargetTrain=target_train, Params=NN_params, Epoch=EPOCH, Lr=LR, Seed=SEED, Eval=True, JointEval=joint_test, ProdEval=prod_test)        
            DI_DV_Eval.append(DI_DV_e)    
        else:   
            model, loss_e = CMINE.train_classifier(BatchTrain=batch_train, TargetTrain=target_train, Params=NN_params, Epoch=EPOCH, Lr=LR, Seed=SEED)
        
        #Compute directed information
        DI_est = CMINE.estimate_CMI(model, joint_test, prod_test)
    
        print('Duration: ', time.time()-start_time, ' seconds')               
        print('DV=',DI_est[1])   
        print('True=',True_DI)        
        DI_DV_t.append(DI_est[1])
        
    DI_DV.append(np.mean(DI_DV_t))
    
file = open('Data/DI/result_DI_'+Link, 'wb')
pickle.dump((True_DI,DI_DV,DI_DV_Eval,n,K,LR,EPOCH,loss_e,params), file)

file.close()    
