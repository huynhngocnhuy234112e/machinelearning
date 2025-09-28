class ListProduct:
    def __init__(self):
        self.products=[]
    def add_product(self,p):
        self.products.append(p)
    def print_products(self):
        for p in self.products:
            print(p)
    def sort_desc_price(self):
        for i in range(0, len(self.products)):
            # 1. Tìm chỉ mục của sản phẩm đắt nhất (max_idx) trong phần còn lại
            max_idx = i
            for j in range(i + 1, len(self.products)):

                # SỬA 1: So sánh thuộc tính 'price'
                if self.products[j].price > self.products[max_idx].price:
                    max_idx = j

            # SỬA 2: Hoán đổi phần tử tại i và max_idx
            # Đây là cách hoán đổi chuẩn trong Python
            self.products[i], self.products[max_idx] = self.products[max_idx], self.products[i]
