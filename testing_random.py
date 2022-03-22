lst = [i for i in range(int(1e6))]
def sample():
    import random
    random.seed(20)
    spl = random.sample(lst,1)
    print(spl)
def main():
    i = 0
    sample()

if __name__ == "__main__":
    main()