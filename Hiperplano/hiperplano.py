import numpy as np
import matplotlib.pyplot as plt
import math

def proyeccion(p_test, p_intermedio, magnitud):
    proy= np.vdot(p_test, p_intermedio)/magnitud
    print("La proyección es de: ", proy)


#positive = np.array([5,7], [6,6], [5,5], [4,5], [4,6])
#negative = np.array([1,1], [2,1], [3,1], [1,2], [2,2])

Xpos = np.array([5, 6, 5, 4, 4])
Ypos = np.array([7, 6, 5, 5, 6])

Xneg= np.array([1, 2, 3, 1, 2])
Yneg= np.array([1, 1, 1, 2, 2])

plt.scatter(Xpos,Ypos)
plt.scatter(Xneg, Yneg)

sum_Xpos=0
sum_Ypos=0

for x in Xpos:
    sum_Xpos=sum_Xpos+x
    
for y in Ypos:    
    sum_Ypos=sum_Ypos+y

cxpos=(1/(len(Xpos)))*(sum_Xpos)
cypos=(1/(len(Ypos)))*(sum_Ypos)

plt.scatter(cxpos, cypos)

sum_Xneg=0
sum_Yneg=0

for x in Xneg:
    sum_Xneg=sum_Xneg+x
    
for y in Yneg:    
    sum_Yneg=sum_Yneg+y

cxneg=(1/(len(Xneg)))*(sum_Xneg)
cyneg=(1/(len(Yneg)))*(sum_Yneg)

plt.scatter(cxneg, cyneg, color="black")

cx= (cxpos+cxneg)/2
cy= (cypos+cyneg)/2

plt.scatter(cx, cy)

cmag= math.sqrt((cx*cx)+(cy*cy))

print('Magnitud de C: ',cmag)
#TEST
X=[3,3]
Y=[2,4]
Z=[4,4]

proyeccion(X, [cx,cy], cmag)
proyeccion(Y, [cx,cy], cmag)
proyeccion(Z, [cx,cy], cmag)



plt.show()
