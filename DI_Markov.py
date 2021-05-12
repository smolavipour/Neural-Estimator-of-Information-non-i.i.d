import time
import numpy as np
import pickle
#from numpy.linalg import det

import CMINE_lib as CMINE

#-----------------------------------------------------------------#    
#--------------- Create the dataset ------------------------------#
#-----------------------------------------------------------------#    

n = int(2e5)

sigma_x = 3
sigma_y = 2
sigma_z = 1
a_1 = 1
a_2 = 2
A = np.asarray([[0,0,0],[0,0,0],[0,0,0]])
B = np.asarray([[0,0,0],[a_1,0,0],[0,a_2,0]])

params = (A,B,sigma_x,sigma_y,sigma_z)

#Estimate I(X->Y) or equivalently I(X_1 X_2 X_3;Y_3|Y_1 Y_2)
Link= 'X_Y'

#True_CMI = 0.5*np.log(1+ (sigma_y**2+sigma_z**2)/(sigma_x**2))
#True_CMI=0
True_DI = 0.5*np.log(1+ a_1**2 * sigma_x**2/sigma_y**2)

K = 2
b_size = n//2

#----------------------------------------------------------------------#
#------------------------Train the network-----------------------------#
#----------------------------------------------------------------------#

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
        
    #Create dataset
    dataset = CMINE.create_dataset(GenModel='Markov_Gaussian_2', Params=params, N=n)
    dataset = CMINE.prepare_dataset(dataset, Mode='norm',S=4)
    #[x,y,z,x_,y_,z_,x__,y__,z__]
    
    arrng = [[0,3,6],[1],[4,7]]

    for t in range(T): 
        start_time = time.time()
        
        batch_train, target_train, joint_test, prod_test = CMINE.batch_construction(data=dataset, arrange=arrng, set_size=b_size, K_neighbor=K)    
        print('Duration of data preparation: ',time.time()-start_time,' seconds')
        
        DI_DV_Eval=[]

        start_time = time.time()
        #Train
        if EVAL:
            model, loss_e, DI_LDR_e, DI_DV_e, DI_NWJ_e = CMINE.train_classifier(BatchTrain=batch_train, TargetTrain=target_train, Params=NN_params, Epoch=EPOCH, Lr=LR, Seed=SEED, Eval=True, JointEval=joint_test, ProdEval=prod_test)        
            DI_DV_Eval.append(DI_DV_e)    
        else:   
            model, loss_e = CMINE.train_classifier(BatchTrain=batch_train, TargetTrain=target_train, Params=NN_params, Epoch=EPOCH, Lr=LR, Seed=SEED)
        
        #Compute I(X_1;Y_1|Z_1)
        DI_est = CMINE.estimate_CMI(model, joint_test, prod_test)
        #print(DI_est)
    
        print('Duration: ', time.time()-start_time, ' seconds')               
        print('DV=',DI_est[1])   
        print('True=',True_DI)        
        DI_DV_t.append(DI_est[1])
        
    DI_DV.append(np.mean(DI_DV_t))
    
file = open('Data/DI/result_DI_'+Link, 'wb')
pickle.dump((True_DI,DI_DV,DI_DV_Eval,n,K,LR,EPOCH,loss_e,params), file)

file.close()    
