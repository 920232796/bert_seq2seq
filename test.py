

class A:
    def __init__(self, aa):
        print(aa)


class B(A):
    def __init__(self, bb):
        super().__init__()
        print(bb)



if __name__ == '__main__':


    b = B(bb="hello")

