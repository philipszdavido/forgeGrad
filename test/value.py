from src.engine.Value import Value

a = Value(2)
b = Value(3)

d = a + b
d.grad = 1.0

d.backward()
a.backward()
b.backward()

print(d)
print(a)
print(b)