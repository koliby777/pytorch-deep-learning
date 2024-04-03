import turtle as t
t.width(width=7)
for delka in range(10, 250, 8):
    for barva in ("blue","red","green"):
        t.color(barva)
        t.forward(delka)
        t.right(86)
t.penup()
t.exitonclick()
