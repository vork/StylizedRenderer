"""Frozen desktop entry point, also accepting the renderer's capture CLI."""
import sys
from inkforge.app import main

if __name__=='__main__':
    if '--output' not in sys.argv and '--viewer' not in sys.argv:
        sys.argv.append('--viewer')
    main()
