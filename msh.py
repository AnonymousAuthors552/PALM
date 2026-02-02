import hashlib
from sympy import nextprime
from secrets import randbelow
import random
import time

#### Configuration
PRIME_BITS = 2048  
g = 2

def hash_to_int(x: bytes) -> int:
    return int.from_bytes(hashlib.sha256(x).digest(), 'big')

def hash_to_group(x: bytes, p: int) -> int:
    return pow(g, hash_to_int(x), p)

### from https://link.springer.com/chapter/10.1007/978-3-540-40061-5_12
class MSetMuHash:
    def __init__(self, p=None, value=1):
        self.p = p or nextprime(2 ** PRIME_BITS)
        self.value = value

    def reset(self, value=1):
        self.value = value

    def add(self, element: str):
        h = hash_to_group(element.encode(), self.p)
        self.value = (self.value * h) % self.p

    def multiply(self, element: str):
        e = int(element)
        self.value = (self.value * e) % self.p

    def remove(self, element: str):
        h = hash_to_group(element.encode(), self.p)
        self.value = (self.value * pow(h, -1, self.p)) % self.p

    def digest(self) -> int:
        return self.value

if __name__ == '__main__':
    size = 100
    list1 = list(range(size))  
    list2 = list1[:]
    random.shuffle(list1)
    random.shuffle(list2)
    if list1 != list2:
        print(f"Two lists of {size} are not equal")
        print("First 10 elements...")
        print(f"\t list1: {list1[:10]}")
        print(f"\t list2: {list2[:10]}\n----------")
        mset = MSetMuHash()
        start = time.perf_counter()
        for elt in list1:
            mset.add(str(elt))
        end = time.perf_counter()            
        
        mset2 = MSetMuHash()
        for elt2 in list2:
            mset2.add(str(elt2))

        d1 = mset.digest()
        d2 = mset2.digest()
        print("Digest 1:\n\t", d1)
        print("Digest 2:\n\t", d2)

        if d1 == d2:
            print("----------\nSuccess -- d1 == d2")
        else:
            print("----------\nFail: d1 != d2")
        print(f"---------\nHashing one list time: {end-start}, per record: {(end-start)/size}")
    else:
        print("Lists are equal")
