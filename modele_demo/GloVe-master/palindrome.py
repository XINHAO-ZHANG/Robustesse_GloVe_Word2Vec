def check(elem):
    return elem == elem[::-1]

nombres = list(range(250))
nombres_ch = list(map(str, nombres))
palindromes = list(filter(check, nombres_ch))
sol = list(map(int, palindromes))
print(sol)