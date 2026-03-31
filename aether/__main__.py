import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    import main as aether_main
    aether_main.run()

if __name__ == '__main__':
    main()
