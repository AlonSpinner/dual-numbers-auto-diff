import math

class Dual:
    def __init__(self, real=0, dual=0):
        self.real = real
        self.dual = dual

    def __add__(self, other):
        if isinstance(other, Dual):
            return Dual(self.real + other.real, self.dual + other.dual)
        return Dual(self.real + other, self.dual)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Dual):
            return Dual(self.real - other.real, self.dual - other.dual)
        return Dual(self.real - other, self.dual)

    def __rsub__(self, other):
        return Dual(other, 0).__sub__(self)
    
    def __neg__(self):
        return Dual(-self.real, -self.dual)

    def __mul__(self, other):
        if isinstance(other, Dual):
            return Dual(self.real * other.real, self.real * other.dual + self.dual * other.real)
        return Dual(self.real * other, self.dual * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, exponent):
        if self.dual == 0:
            return Dual(self.real ** exponent, 0)
        
        real_part = self.real ** exponent
        dual_part = exponent * (self.real ** (exponent - 1)) * self.dual
        return Dual(real_part, dual_part)

    def __rpow__(self, base):
        if self.dual == 0:
            return Dual(base ** self.real, 0)

        real_part = base ** self.real
        dual_part = real_part * math.log(base) * self.dual
        return Dual(real_part, dual_part)

    def __truediv__(self, other):
        if isinstance(other, Dual):
            real_part = self.real / other.real
            dual_part = (self.dual - real_part * other.dual) / other.real
            return Dual(real_part, dual_part)
        return Dual(self.real / other, self.dual / other)

    def __rtruediv__(self, other):
        return Dual(other, 0).__truediv__(self)

    def __repr__(self):
        return f"{self.real} + {self.dual}ε"

    def __eq__(self, other):
        if isinstance(other, Dual):
            return self.real == other.real and self.dual == other.dual
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def sin(self):
        real_part = math.sin(self.real)
        dual_part = math.cos(self.real) * self.dual
        return Dual(real_part, dual_part)

    def cos(self):
        real_part = math.cos(self.real)
        dual_part = -math.sin(self.real) * self.dual
        return Dual(real_part, dual_part)

    def tan(self):
        real_part = math.tan(self.real)
        dual_part = (1 + real_part ** 2) * self.dual
        return Dual(real_part, dual_part)

    def exp(self):
        real_part = math.exp(self.real)
        dual_part = real_part * self.dual
        return Dual(real_part, dual_part)

    def log(self):
        if self.real <= 0:
            raise ValueError("Real part must be greater than zero for logarithm function.")
        real_part = math.log(self.real)
        dual_part = self.dual / self.real
        return Dual(real_part, dual_part)

    def sqrt(self):
        if self.real < 0:
            raise ValueError("Real part must be greater than or equal to zero for square root function.")
        real_part = math.sqrt(self.real)
        dual_part = self.dual / (2 * real_part)
        return Dual(real_part, dual_part)
