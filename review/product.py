class Product:
    def __init__(self,id=None,name=None,quantity=None,price=None):
        self.id=id
        self.name=name
        self.quantity=quantity
        self.price=price
    def __str__(self):
        infor="{}\t{}\t{}\t{}".format(self.id,self.name,self.quantity,self.price)
        return infor
