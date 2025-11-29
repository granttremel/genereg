
import random

def main():

    from genereg import draw
    # draw.show_colors()
    # draw.show_colors_24b(0, 100, 1)
    
    step = 8
    
    while True:
        cmin = random.randint(0, 2**24 - step*128)
        cmax = cmin + step*128
        draw.show_colors_24b(cmin, cmax, step)
        input()



if __name__=="__main__":
    main()


