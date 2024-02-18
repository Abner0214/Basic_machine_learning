import torch
from torch.autograd import Variable
import torch.optim as optim

def find_min_optimizer(a4, a3, a2, a1, a0):
    x = Variable(torch.tensor([0.0]), requires_grad=True)
    optimizer = optim.Adam([x], lr=0.01)

    for _ in range(5000):  # You may adjust the number of iterations
        y = a4 * x**4 + a3 * x**3 + a2 * x**2 + a1 * x + a0
        optimizer.zero_grad()
        y.backward()
        optimizer.step()

    xmin = x.item()
    return xmin

def find_min_recursive(a4, a3, a2, a1, a0, x=0.0, lr=0.0001, num_iterations=5000):
    for _ in range(num_iterations):
        gradient = 4 * a4 * x**3 + 3 * a3 * x**2 + 2 * a2 * x + a1
        x = x - lr * gradient

    xmin = x
    return xmin


# User input for coefficients
a4 = float(input("Enter the coefficient a4 (> 0): "))
a3 = float(input("Enter the coefficient a3: "))
a2 = float(input("Enter the coefficient a2: "))
a1 = float(input("Enter the coefficient a1: "))
a0 = float(input("Enter the coefficient a0: "))

# Call the functions
result_optimizer = find_min_optimizer(a4, a3, a2, a1, a0)
result_recursive = find_min_recursive(a4, a3, a2, a1, a0)

# Display the results
print("Minimum using optimizer:", result_optimizer)
print("Minimum using recursive method:", result_recursive)
