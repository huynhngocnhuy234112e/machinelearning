def sum_1_n(n):
    s=0
    for i in range(1,n+1):
        s=s+i
    return s
print(sum_1_n(5))
def sumfx(x,n):
    s=0
    for i in range(1,n+1):
        s=s+x**i/sum_1_n(i)
    return s
x=2
n=3
r=sumfx(x,n)
print("S({},{})={}".format(x,n,r))
















































































