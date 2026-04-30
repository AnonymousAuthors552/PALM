import hashlib
from sympy import nextprime
import random
import time

from ecdsa import NIST256p, ellipticcurve, numbertheory

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
        self.value = int(value)

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

class ECMH:
    def __init__(self, curve=NIST256p, value=None):
        self.curve = curve
        self.G = curve.generator
        self.n = curve.order
        self.p = curve.curve.p()
        self.a = curve.curve.a()
        self.b = curve.curve.b()
        self.value = value

    def reset(self, value=None):
        if value is None:
            self.value = None
        else:
            self.value = self._point_from_digest(int(value))

    def _hash_to_point(self, data: bytes):
        """Try-and-increment: hash bytes > curve point directly."""
        counter = 0
        while True:
            attempt = hashlib.sha256(data + counter.to_bytes(4, 'big')).digest()
            x = int.from_bytes(attempt, 'big') % self.p
            # y² = x³ + ax + b  (mod p)
            rhs = (pow(x, 3, self.p) + self.a * x + self.b) % self.p
            # check if rhs is a quadratic residue
            if pow(rhs, (self.p - 1) // 2, self.p) == 1:
                y = numbertheory.square_root_mod_prime(rhs, self.p)
                return ellipticcurve.PointJacobi(
                    self.curve.curve, x, y, 1, self.n
                )
            counter += 1

    def add(self, element: str):
        P = self._hash_to_point(element.encode())
        self.value = P if self.value is None else self.value + P

    def multiply(self, digest_int):
        """Merge a worker's digest int into current state via point addition."""
        P = self._point_from_digest(int(digest_int))
        if P is None:
            return
        self.value = P if self.value is None else self.value + P

    def remove(self, element: str):
        P = self._hash_to_point(element.encode())
        negP = ellipticcurve.PointJacobi(
            self.curve.curve, P.x(), (-P.y()) % self.p, 1, self.n
        )
        self.value = negP if self.value is None else self.value + negP

    def digest(self) -> int:
        if self.value is None:
            return 0
        x = self.value.x()
        y = self.value.y()
        prefix = b'\x02' if y % 2 == 0 else b'\x03'
        compressed = prefix + x.to_bytes(32, 'big')
        return int.from_bytes(compressed, 'big')
    
    def _point_from_digest(self, digest_int: int):
        """Reconstruct point from compressed digest int."""
        if digest_int == 0:
            return None
        digest_bytes = digest_int.to_bytes(33, 'big')
        prefix = digest_bytes[0]
        x = int.from_bytes(digest_bytes[1:], 'big')
        rhs = (pow(x, 3, self.p) + self.a * x + self.b) % self.p
        y = numbertheory.square_root_mod_prime(rhs, self.p)
        if (y % 2) != (prefix & 1):
            y = self.p - y
        return ellipticcurve.PointJacobi(self.curve.curve, x, y, 1, self.n)

    def equal(self, other: 'ECMH') -> bool:
        return self.digest() == other.digest()


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
        # list1[11] = 10
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
