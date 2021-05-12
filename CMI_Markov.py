import time
import numpy as np
import pickle

import CMINE_lib as CMINE

cntr=0
sigma_v = np.arange(0.05,3,0.1)
for sigma_x in sigma_v:
    cntr+=1
    #-----------------------------------------------------------------#    
    #--------------- Create the dataset ------------------------------#
    #-----------------------------------------------------------------#    
    
    n = int(2e4)

    sigma_y = 2
    sigma_z = 2
    arrng = [[0],[1],[]]
    A = np.asarray([[0,1,0],[0,0,0],[0,0,0]])
    B = np.asarray([[0,1,0],[0,0,1],[0,0,0]])
    d=10    
    
    params = (A,B,sigma_x,sigma_y,sigma_z)
    True_CMI = d*0.5*np.log(1+ (sigma_y**2+sigma_z**2)/(sigma_x**2 + sigma_y**2 + sigma_z**2))
    print("True_CMI:",True_CMI)
    
    K = 2
    b_size = n//2
    
    #----------------------------------------------------------------------#
    #------------------------Train the network-----------------------------#
    #----------------------------------------------------------------------#
    
    # Set up neural network paramters
    LR = 1e-3
    EPOCH = 200
    SEED = 123
    input_size = 2*d
    hidden_size = 64
    num_classes = 2
    tau = 1e-3
    
    NN_params = (input_size,hidden_size,num_classes,tau)
    EVAL = False
    
    #Monte Carlo param
    T = 20
    S = 10
    
    
    CMI_DV = []
    
    for s in range(S):
        CMI_DV_t = []
            
        #Create dataset
        dataset = CMINE.create_dataset(GenModel='Markov_Gaussian_1', Params=params, Dim=d, N=n)
        dataset = CMINE.prepare_dataset(dataset, Mode='norm',S=4)
    
        for t in range(T): 
            start_time = time.time()
            
            batch_train, target_train, joint_test, prod_test = CMINE.batch_construction(data=dataset, arrange=arrng, set_size=b_size, K_neighbor=K)    
            print('Duration of data preparation: ',time.time()-start_time,' seconds')
            
            CMI_DV_Eval=[]
    
            start_time = time.time()
            #Train
            if EVAL:
                model, loss_e, CMI_LDR_e, CMI_DV_e, CMI_NWJ_e = CMINE.train_classifier(BatchTrain=batch_train, TargetTrain=target_train, Params=NN_params, Epoch=EPOCH, Lr=LR, Seed=SEED, Eval=True, JointEval=joint_test, ProdEval=prod_test)        
                CMI_DV_Eval.append(CMI_DV_e)    
            else:   
                model, loss_e = CMINE.train_classifier(BatchTrain=batch_train, TargetTrain=target_train, Params=NN_params, Epoch=EPOCH, Lr=LR, Seed=SEED)
            
            #Compute I(X_1;Y_1|Z_1)
            CMI_est = CMINE.estimate_CMI(model, joint_test, prod_test)
            print(CMI_est)
        
            print('Duration: ', time.time()-start_time, ' seconds')               
            print('DV=',CMI_est[1])   
            print('True=',True_CMI)        
            CMI_DV_t.append(CMI_est[1])
            
        CMI_DV.append(np.mean(CMI_DV_t))
        
    file = open('Data/CMI/result_d_'+str(cntr), 'wb')
    pickle.dump((True_CMI,CMI_DV,CMI_DV_Eval,n,K,LR,EPOCH,loss_e,params), file)

file.close()    
