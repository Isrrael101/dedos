import turtle
import math

# Set up the screen
screen = turtle.Screen()
screen.bgcolor("black")
screen.title("Spirograph with Name")

# Create and configure the turtle
t = turtle.Turtle()
t.speed(0)  # Fastest speed
t.pensize(2)

# Function to create spirograph pattern
def draw_spirograph(R, r, d):
    t.penup()
    t.goto(0, -200)  # Start position
    t.pendown()
    
    # Create colorful pattern
    for angle in range(360):
        t.pencolor("#%06x" % (int(angle * 720) % 16777215))  # Rainbow effect
        theta = math.radians(angle)
        x = (R - r) * math.cos(theta) + d * math.cos((R - r) * theta / r)
        y = (R - r) * math.sin(theta) - d * math.sin((R - r) * theta / r)
        t.goto(x, y)

# Draw the spirograph
t.penup()
draw_spirograph(100, 50, 50)

# Write the name
t.penup()
t.goto(0, 0)
t.color("white")
style = ("Arial", 36, "bold")
t.write("Isrrael", align="center", font=style)

# Hide the turtle and keep the window open
t.hideturtle()
screen.mainloop()