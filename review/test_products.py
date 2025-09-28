from product import Product
from products import ListProduct

lp=ListProduct()
lp.add_product(Product(id="p1",name="coca",quantity=15,price=35))
lp.add_product(Product(id="p2",name="pepsi",quantity=14,price=25))
lp.add_product(Product(id="p3",name="sting",quantity=20,price=32))
lp.add_product(Product(id="p4",name="redbull",quantity=30,price=25))
lp.print_products()
lp.sort_desc_price()
print("--List Products")
lp.print_products()