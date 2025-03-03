x = 10  # Global x

def outer():
    x = 28  # Local x in outer function

    def inner():
        global x  # Declare x as global inside inner function
        x = 30   # Modify the global x

    inner()
    print('Outer', x)  # This will print the local x (28)

outer()
print('Global', x)  # This will print the global x (which was modified to 30)