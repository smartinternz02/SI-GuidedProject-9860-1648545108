#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install numpy')


# In[4]:


get_ipython().system('pip install tensorflow')


# In[5]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[6]:


train_datagen = ImageDataGenerator(rescale=1/255,zoom_range=0.2,horizontal_flip=True,vertical_flip=False)


# In[7]:


train_datagen = ImageDataGenerator(rescale=1/255)


# In[8]:


test_datagen = ImageDataGenerator(rescale=1/255)


# In[9]:


x_train=train_datagen.flow_from_directory(r'C:\Users\Asmi Bhardwaj\Downloads\archive\Cars Dataset\train',target_size=(64,64),class_mode = 'categorical',batch_size=100)


# In[10]:


len(x_train)


# In[11]:


x_test=test_datagen.flow_from_directory(r'C:\Users\Asmi Bhardwaj\Downloads\archive\Cars Dataset\test',target_size=(64,64),class_mode = 'categorical',batch_size=100)


# In[12]:


len(x_test)


# In[13]:


x_train.class_indices


# # Importing Libraries

# In[14]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense


# # Create the Model

# In[15]:


model = Sequential()


# 
# # Adding Layers

# In[16]:


model.add(Convolution2D(32,(3,3),activation='relu',input_shape=(64,64,3)))


# In[17]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[18]:


model.add(Flatten())


# In[19]:


#hidden layer - 1
model.add(Dense(300,activation = 'relu'))


# In[20]:


#hiddenlayer - 2
model.add(Dense(150,activation='relu'))


# In[21]:


#output layer
model.add(Dense(7,activation='softmax'))


# # Compile the Model

# In[22]:


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# # Fit The Model

# In[23]:


#model.fit_generator(x_train,steps_per_epoch = len(x_train),epochs=10,validation_data=x_test,validation_steps=len(x_test))

#model.fit_generator(x_train,steps_per_epoch = len(x_train),epochs=10,validation_data = x_test,validation_steps=len(x_test))

#model.fit_generator(x_train,steps_per_epoch=len(x_train),epochs=10)

model.fit_generator(x_train,steps_per_epoch=len(x_train),epochs=10,validation_data=x_test,validation_steps=len(x_test))


# In[24]:


model.save('cars.h5')


# # Testing the CNN model

# In[25]:


import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# In[26]:


model = load_model('cars.h5')


# In[27]:


img = image.load_img(r'C:\Users\Asmi Bhardwaj\Downloads\archive\Cars Dataset\test\Rolls Royce\239.jpg',target_size=(64,64))


# In[28]:


img


# In[29]:


x=image.img_to_array(img)


# In[30]:


x.ndim


# In[31]:


x=np.expand_dims(x,axis=0)


# In[32]:


x


# In[33]:


x.ndim


# In[34]:


pred=np.argmax(model.predict(x),axis=1)


# In[35]:


pred


# In[36]:


index = ['Audi','Hyundai Creta','Mahindra Scorpio','Rolls Royce','Swift','Tata Safari','Toyota Innova']
print(index[pred[0]])


# # Open CV

# In[41]:


import cv2


# In[42]:


img = cv2.imread(r'C:\Users\Asmi Bhardwaj\Downloads\archive\Cars Dataset\test\Rolls Royce\239.jpg',1)


# In[43]:


img


# In[44]:


img1 = cv2.imread(r'C:\Users\Asmi Bhardwaj\Downloads\archive\Cars Dataset\test\Rolls Royce\239.jpg',0)


# In[45]:


img1


# In[46]:


print(img.shape)


# In[ ]:


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # CNN Video Analysis

# In[ ]:


import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
model=load_model('cars.h5')
video=cv2.VideoCapture(0)
index=['Audi','Hyundai Creta','Mahindra Scorpio','Rolls Royce','Swift','Tata Safari','Toyota Innova']
while 1:
    succes,frame=video.read()
    cv2.imwrite('image.jpg',frame)
    img=image.load_img('image.jpg',target_size=(64,64))
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    pred=np.argmax(model.predict(x),axis=1)
    y=pred[0]
    cv2.putText(frame,'The predicted Cars is: '+str(index[y]),(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),4)
    cv2.imshow('image',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()


# In[ ]:




