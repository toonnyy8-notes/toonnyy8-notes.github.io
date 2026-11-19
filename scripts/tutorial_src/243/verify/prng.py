"""mulberry32 + Box-Muller (cos branch), bit-identical to the JS the page will use:
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);
 t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;}}
gauss(r) = sqrt(-2 ln(1-u1)) cos(2 pi u2)"""
import math
M32 = 0xFFFFFFFF
def imul(x, y): return (x * y) & M32
def mulberry32(a):
    st = [a & M32]
    def r():
        a = (st[0] + 0x6D2B79F5) & M32; st[0] = a
        t = imul(a ^ (a >> 15), a | 1)
        t = ((t + imul(t ^ (t >> 7), t | 61)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296
    return r
def gauss(r):
    u1 = r(); u2 = r()
    return math.sqrt(-2 * math.log(1 - u1)) * math.cos(2 * math.pi * u2)
if __name__ == '__main__':
    r = mulberry32(7); print([round(r(), 10) for _ in range(3)])
