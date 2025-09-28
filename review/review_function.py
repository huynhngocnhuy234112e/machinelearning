def digit_separator(n):
    n1=n%10
    n2=(n//10)%10
    n3=n//100
    return n1,n2,n3
n1,n2,n3=digit_separator(789)
print("n1=",n1)
print("n2=",n2)
print("n3=",n3)

