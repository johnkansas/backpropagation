import numpy as np

def f(g):
    return g*3

def g(x):
    return x+1

x=3
F=f(g(x))
print(F)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def MSE(targets,values):
    check=isinstance(values,list)
    if not check:
        print(check)
        return False
    result =0
    for i,target in enumerate(targets):
        result+= 0.5*(target-values[i])**2

    return result

MSE([0.2,0.7],[0.57,0.61])

#input
x1=0.2
x2=0.5

#target
t1=0.2
t2=0.7

w0=[[0.1,0.2],[0.3,0.1]]
w1=[[0.4,0.5],[0.1,0.3]]

learningrate =0.3
limit =1000

def updateSecondLayerWeight(targetY,y,prevY,updatedWeight):
    v1=-(targetY-y)+0
    v2=y*(1-y)
    deff=v1*v2*prevY
    return updatedWeight-learningrate*deff

def updateFirstLayerWeight(t1,t2,y1,y2,w1,w2,a,updateWeight):
    e1=-(t1-y1)*y1*(1-y1)*w1
    e2=-(t2-y2)*y2*(1-y2)*w2
    v1=a*(1-a)
    v2=a
    deff=(e1+e2)*v1*v2

    return updateWeight-learningrate*deff

for i in range(0,limit):
    z10=x1*w0[0][0]+x2*w0[1][0]
    a10=sigmoid(z10)
    z11=x1*w0[0][1]+x2*w0[1][1]
    a11=sigmoid(z11)

    z20=a10*w1[0][0]+a11*w1[1][0]
    a20=sigmoid(z20)
    z21=a10*w1[0][1]+a11*w1[1][1]
    a21=sigmoid(z21)

    e_t=MSE([t1,t2],[a20,a21])

    print("i=",i,"y1=",a20,"y2=",a21,"E=",e_t)

    neww0=[[
        updateFirstLayerWeight(t1,t2,a20,a21,w1[0][0],w1[0][1],a10,w0[0][0]),
        updateFirstLayerWeight(t1,t2,a20,a21,w1[1][0],w1[1][1],a11,w0[0][1])
    ],
    [
        updateFirstLayerWeight(t1,t2,a20,a21,w1[0][0],w1[0][1],a10,w0[1][0]),
        updateFirstLayerWeight(t1,t2,a20,a21,w1[1][0],w1[1][1],a11,w0[1][1])
    ]]

    neww1=[
        [updateSecondLayerWeight(t1,a20,a10,w1[0][0]),
        updateSecondLayerWeight(t1,a21,a10,w1[0][1])],
        [updateSecondLayerWeight(t1,a20,a11,w1[1][0]),
        updateSecondLayerWeight(t1,a21,a11,w1[1][1])]
    ]
    for i,v in enumerate(neww0):
        for ii,vv in enumerate(v):
            w0[i][ii]=vv
    for i,v in enumerate(neww1):
        for ii,vv in enumerate(v):
            w1[i][ii]=vv

print("t1=",t1,"t2=",t2)