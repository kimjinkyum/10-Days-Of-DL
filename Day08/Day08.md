## activation function

###  1. Implement hardware (simple model)

- AND/OR problem- linearly separable O

- XOR problem - linearly separable X

 : solve <b>Backpropagation</b>

###  2. Backpropagation

#### Problem

1) Backpropagation not work well for with many layers

2) other rising machine learning algorithms : SVM, randomforeset

## Lab


<b>ndim(rank),shape,axis</b>
    
    t= tf.constant([1,2,3,4])
    t.ndim           #dimesion print 1
    shape            #print[4]
    
    t1 = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
    t1.ndim          #print 4
    t1.shape         #print [1,2,3,4]
    
   <img src="https://user-images.githubusercontent.com/44569994/68076070-93333e00-fdf3-11e9-81fb-a531ae3d35a7.png" width="70%">
   
    axis in 2-demensional
    axis=0 : row
    axis=1 :column


<b>broadcasting</b>
    
    Automatically calculate even when it doesn't match shape


<b>tf.reduce_mean</b>

        tf.reduce_mean([1,2])
        ** print 1 (not 1.5)
        
        using
        tf.reduce_mean([1.0,2.0])


<b>shape</b>
    
    reshape : transform shpae
    notation -1  : auto
    squeeze : delete one dimension
    expand_dim : add dimension at axis(input)
    
    tf.squeeze(t1).shape : [2,3,4]
    
    ** only uses axis shape 1 
    t2=tf.constant([[[[1, 2, 3, 4]],[ [21, 22, 23, 24]]]])  #shape [1,2,1,4]
    tf.squeeze(t2,[2]).shape   #shape [1,2,4]
    tf.squeeze(t2,[4]).shape   #Error!
    
    tf.expand_dim(t2,[1]).shpae  #shape(1,1,2,1,4)
    tf.expand_dim(t2,[5]).shpae  #Error!


