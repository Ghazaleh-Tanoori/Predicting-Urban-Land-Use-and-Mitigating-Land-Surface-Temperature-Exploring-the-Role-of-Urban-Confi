import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage import io
#from kerastuner import HyperParameter, HyperParameters
#from kerastuner.tuners import RandomSearch, Hyperband
import pdb
#------------------------------------------------------------------
#---------------------------Set CNN model Paramters
def Set_paramter_NN_model(hp):
  num_filters=hp.Choice('num_filter',values=[2])
  num_cnn=hp.Choice('num_Conv',values=[1,2,3,4])
  K=hp.Choice('kernel_size',values=[2,3])
  active1=hp.Choice('active1',values=['relu','linear','selu'])
  active2=hp.Choice('active2',values=['relu','linear','selu'])
  DP=hp.Choice('dropout',values=[0.2,0.4,0.5,0.6,0.7])
  #-------------------------------------------
   # first layer
  Input1=tf.keras.layers.Input(shape=(image_X,image_Y,3,1)) 
  X1=tf.keras.layers.Conv3D(filters=num_filters,kernel_size=(K,K,K),
                                 padding='same',
                                 activation=active1,name='Conv_0')(Input1)
  num_filters=num_filters*2
  # other layers
  for i in range(1,num_cnn):
    X1=tf.keras.layers.Conv3D(filters=num_filters,kernel_size=(K,K,K),
                                 padding='same',name='Conv_'+str(i))(X1)
    X1=tf.keras.layers.Dropout(rate=DP)(X1)
    X1=tf.keras.layers.Add()([Input1,X1])
    X1=tf.keras.layers.Activation(active1)(X1)
    #mdl.add(tf.keras.layers.MaxPool3D(pool_size=(2,2,1)))
    num_filters=num_filters*2
  
  # output
  #X1=tf.keras.layers.Add()([Input1,X1])
  prediction=tf.keras.layers.Conv3D(filters=1,kernel_size=(K,K,K),
                                 padding='same',activation=active2)(X1)
 
  #-------------------------------------------
  #Lrate=hp.Choice('learning_rate',values=[0.001,0.005,0.01])
  mdl=tf.keras.Model(inputs=[Input1],outputs=[prediction])
  mdl.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mse',"accuracy"])
  return mdl
#------------------------------------------------------------------
def set_param(X_Train,Y_Train,X_val,Y_val):
  # find best Prot parameters
    print('===============Run set Hyperparam==============')
    tuner2=RandomSearch(Set_paramter_NN_model,objective="val_mse",max_trials=50,overwrite=True)
    tuner2.search(X_Train,Y_Train,epochs=5,validation_data=(X_val,Y_val))
    #--------------------------------------------
    print("***********************Best HyperParameters*********************************")
    best_hps=tuner2.get_best_hyperparameters()[0]
    print(best_hps.values)
    #----------------------------------------------------
    pdb.set_trace()
    num_conn=best_hps['num_Conv']
    num_filter=best_hps['num_filter']
    kernel_size=best_hps['kernel_size']
    active1=best_hps['active1']
    active2=best_hps['active2']
    Drop_out=best_hps['dropout']
    Lrate=best_hps['learning_rate']
    net_info=net(num_conn,num_filter,kernel_size,active1,active2,Drop_out,Lrate)
    

    pdb.set_trace()
    return net_info

#------------------------------------------------------------------
class net:
    def __init__(self,num_con,num_filter,kernel_size,active1,active2,Drop_out,Lrate):
        self.num_con=num_con
        self.num_filter=num_filter
        self.kernel_size=kernel_size
        self.active1=active1
        self.active2=active2
        self.Drop_out=Drop_out
        self.Lrate=Lrate
        
#--------------------------------------------------------------------
#------------------------------------------------------------------
#-------------------------Residual prediction
def NN_model_updown_sample(net_info,image_X,image_Y):
  Input1=tf.keras.layers.Input(shape=(image_X,image_Y,3,1))
  X1=Input1
  num_filters=net_info.num_filter
  K=net_info.kernel_size
  num_cnn=net_info.num_con
  #----------------------------------
  # first layer
  for i in range(num_cnn):
    X1=tf.keras.layers.Conv3D(filters=num_filters,kernel_size=(K,K,K),
                              padding='same',
                              activation=net_info.active1,name='Conv_0_'+str(i))(X1)
    #X1=tf.keras.layers.Dropout(net_info.Drop_out)(X1)                                
  
  X11=tf.keras.layers.MaxPool3D(pool_size=(2,2,1))(X1)
  # second
  num_filters=num_filters*2
  X2=X11
  for i in range(num_cnn):
    X2=tf.keras.layers.Conv3D(filters=num_filters,kernel_size=(K,K,K),
                              padding='same',
                              activation=net_info.active1,name='Conv_1_'+str(i))(X2) 
    #X2=tf.keras.layers.Dropout(net_info.Drop_out)(X2) 
  X2=tf.keras.layers.MaxPool3D(pool_size=(2,2,1))(X2)                                                       
  ## third
  num_filters=num_filters*2
  X3=X2
  for i in range(num_cnn):
    X3=tf.keras.layers.Conv3D(filters=num_filters,kernel_size=(K,K,K),
                                 padding='same',
                                 activation=net_info.active1,name='Conv_2_'+str(i))(X3)
    #X3=tf.keras.layers.Dropout(net_info.Drop_out)(X3) 
  X3=tf.keras.layers.MaxPool3D(pool_size=(2,2,1))(X3)                                                        
  #----------------------------------------------------------
  num_filters=num_filters/2
  X3=tf.keras.layers.Conv3D(filters=num_filters,kernel_size=(K,K,K),
                                 padding='same',
                                 activation=net_info.active1,name='Conv_3')(X3)
  #upsampling 1
  XD1=tf.keras.layers.UpSampling3D((2, 2,1))(X3)
  XD1=tf.keras.layers.Add()([X2,XD1])
  for i in range(num_cnn-1):
    XD1=tf.keras.layers.Conv3D(filters=num_filters,kernel_size=(K,K,K),padding='same',
                           activation=net_info.active1,name='Conv_up_0'+str(i))(XD1)
                                                      
  
  XD1=tf.keras.layers.UpSampling3D((2, 2,1))(XD1)
  num_filters=num_filters/2
  XD1=tf.keras.layers.Conv3D(filters=num_filters,kernel_size=(K,K,K),
                                 padding='same',
                                 activation=net_info.active1,name='Conv_4_')(XD1)
  XD2=tf.keras.layers.Add()([XD1,X11])
  for i in range(num_cnn-1):
    XD2=tf.keras.layers.Conv3D(filters=num_filters,kernel_size=(K,K,K),padding='same',
                           activation=net_info.active1,name='Conv_up_1'+str(i))(XD2)
  
  XD2=tf.keras.layers.UpSampling3D((2, 2,1))(XD2)
  XD3=tf.keras.layers.Add()([XD2,X1])
  #crop
  #X_Crop=tf.keras.layers.Cropping3D(cropping=((4,3),(3,2),(0,0)))(XD5)
  # output
  prediction=tf.keras.layers.Conv3D(filters=1,kernel_size=(K,K,K),
                                 padding='same',activation=net_info.active2)(XD3)
 
  prediction=tf.keras.layers.Add()([prediction, Input1])
  prediction=tf.keras.layers.Conv3D(filters=1,kernel_size=(K,K,K),
                                 padding='same',activation=net_info.active2)(prediction)
  # model
  mdl=tf.keras.Model(inputs=[Input1],outputs=[prediction])
  mdl.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=net_info.Lrate),
              loss=tf.keras.losses.MeanAbsoluteError(),
              metrics=['accuracy'])
  mdl.summary()
  return mdl  
#------------------------------------------------------------------
#--------------------------Leave one out
def LOO(net_info,X,Y):
  sh=np.shape(X)
  N_tr=sh[0]
  err=[]
  pdb.set_trace()
  for i in range(N_tr):
    X_tr=[]
    Y_tr=[]
    for j in range(N_tr):
      if (i==j):
        X_te=X[j]
        Y_te=Y[j]
        X_te=np.reshape(X_te,(1,image_X,image_Y,3,1))
        Y_te=np.reshape(Y_te,(1,image_X,image_Y,3,1))
      else:
        X_tr.append(X[j])
        Y_tr.append(Y[j])
    X_tr=np.array(X_tr)
    Y_tr=np.array(Y_tr)
    X_te=np.array(X_te)
    Y_te=np.array(Y_te)
    mdl=NN_model_updown_sample(net_info,image_X,image_Y)
    hist=mdl.fit(X_tr,Y_tr,epochs=100,verbose=1)
    Y_pre=mdl.predict(X_te)
    print('shape test',np.shape(Y_pre))
    print('shape true',np.shape(Y_te))
    err.append(np.mean((Y_te-Y_pre)**2))
    Y_pre=np.reshape(Y_te,(image_X,image_Y,3))
    print('new shape',np.shape(Y_pre))
    Y_pre=Y_pre*255.0
    Y_pre=Y_pre.astype(np.uint8)
    f_name1="IM_predict"+str(i)+".tif"
    io.imsave(fname=f_name1, arr=Y_pre)
    print(err)
    pdb.set_trace()

  print('--------------------Leave One out error----------------------------')
  print(err)
  err=np.array(err)
  Avg=np.mean(err)
  print('Avg leave one out:',Avg)  

#------------------------------------------------------------------
#------------------------------------------------------------------
#------------------------------------------------------------------
#------------------------------------------------------------------
#----------------Create Traing Dataset
f_name='IM_'
Years=[2006,2010,2014,2018,2021]
X_Train=[]
Y_Train=[]
image_X=0
image_Y=0
print('--------------Training Set-------------')
for i in range(4):
  # X
  f_name1=f_name+str(Years[i])+'.tif'
  img = io.imread(f_name1)
  img=img/255.0
  img = tf.image.crop_to_bounding_box(img,0,320,656,600)
  #plt.pyplot.imshow(img)
  #print(f_name1)
  #print(np.shape(img))
  #print(f_name1)
  #print(img)
  X_Train.append(img)
  
  #label
  f_name2=f_name+str(Years[i+1])+'.tif'
  img=io.imread(f_name2)#plt.image.imread(f_name2)
  img=img/255.0
  img = tf.image.crop_to_bounding_box(img,0,320,656,600)
  Y_Train.append(img)
  #print(f_name2)
  #print(np.shape(img))
  print('Sequences=[',f_name1,']------------->','Predicted =',f_name2)
  #print(img)

sh=img.shape
print(sh)
image_X=sh[0]
image_Y=sh[1] 
#print(image_X,image_Y)
#print(np.shape(X_Train)) 
print(np.shape(X_Train))
fig=plt.pyplot.figure(figsize=(12,6))  
i=1
for item in X_Train:
  ax=fig.add_subplot(1,4,i)
  ax.imshow(item)
  i=i+1
pdb.set_trace()
#X_Train=np.array(X_Train)
X_Train=np.reshape(X_Train,(4,image_X,image_Y,3,1))
Y_Train=np.array(Y_Train)
Y_Train=np.reshape(Y_Train,(4,image_X,image_Y,3,1))
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-SET PARAMETERS
print('***********************Setp1:set param****************************')
X_tr=X_Train[0:3]
Y_tr=Y_Train[0:3]
X_val=X_Train[3]
Y_val=Y_Train[3]
X_val=np.reshape(X_val,(1,image_X,image_Y,3,1))
Y_val=np.reshape(Y_val,(1,image_X,image_Y,3,1))
pdb.set_trace()
#num_con,num_filter,kernel_size,active1,active2,Drop_out,Lrate
#net_info=set_param(X_tr,Y_tr,X_val,Y_val)
net_info=net(3,16,2,'relu','relu',0.7,0.005)
#------------------------------------------------------------------
#------------------------------------------------------------------
#-Leave one out error
#print('***********************Setp2:Leave one out error****************************')
#LOO(net_info,X_Train,Y_Train)
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-Predicted 2025
print('***********************Setp3:predict 2025****************************')
print('--------Train set size--------')  
print(np.shape(X_Train))
print('------Image Size-------')
print(np.shape(img))
#print(img)
#--------------------Creat test Dataset
X_Test=[]
img=io.imread('IM_2021.tif')#plt.image.imread('IM_2021.jpg')
img=img/255.0
img = tf.image.crop_to_bounding_box(img,0,320,656,600)
X_Test.append(img)
X_Test=np.array(X_Test)
X_Test=np.reshape(X_Test,(1,image_X,image_Y,3,1))
print('-------------Test Set-----------')
print('Sequences={im_2021}')

#------------------Create Model
mdl=NN_model_updown_sample(net_info,image_X,image_Y)
pdb.set_trace()
#------------------Fit Model
hist=mdl.fit(X_Train,Y_Train,epochs=60,verbose=1)
h=hist.history
fig1 = plt.pyplot.figure(figsize=(12, 4))
plt.pyplot.plot(h['loss'])
#-----------------------Predict
print("--------------------------Test Image 2021--------------------------------")
fig=plt.pyplot.figure(figsize=(12,6))
ax=fig.add_subplot(1,4,1)
ax.imshow(img)
ax.set_title('Test_image 2021')
Y_Test=mdl.predict(X_Test)
print('Size Prediction',np.shape(Y_Test))
print("----------------------------------------------------------")
Y_Test=np.reshape(Y_Test,(image_X,image_Y,3))
m1=np.max(Y_Test)
Y_Test=Y_Test/m1
Y_Test=Y_Test*255.0
io.imsave(fname="IM_2025_beforunit.tif", arr=Y_Test)
print(Y_Test)
Y_Test1=Y_Test.astype(np.uint8)
io.imsave(fname="IM_2025.tif", arr=Y_Test1)
ax=fig.add_subplot(1,4,2)
#-------------colors
'''cdict1 = {
    'red': (
        (1.0, 0.0, 0.0),
    ),
    'green': (
        (0.0, 1.0, 0.0),
    ),
    'blue': (
        (0.0, 0.0, 1.0),
    )
}
blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)'''
#--------------------
ax.imshow(Y_Test1)
ax.set_title('Predicted Bluered image 2025')

ax=fig.add_subplot(1,4,3)
ax.imshow(Y_Test1)
ax.set_title('Predicted image 2025')

ax=fig.add_subplot(1,4,4)
ax.imshow(Y_Test)
ax.set_title('Predicted image not unit8 2025')
