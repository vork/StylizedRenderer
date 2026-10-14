"""Exercise real swapchain/UI resizing, not just offscreen target allocation."""
from inkforge.procedural import showroom
from inkforge.renderer import Renderer,Style
from inkforge.viewer import Viewer


def main():
    viewer=Viewer(Renderer(showroom(),960,640,shadow_size=512),[4,4,-7],[0,.5,1],Style(),55)
    try:
        for width,height in ((960,640),(1216,760),(800,600)):
            viewer.window.resize(width,height)
            for _ in range(12):assert viewer.frame()
            actual=(viewer.renderer.width,viewer.renderer.height)
            assert actual==(width,height),(actual,(width,height))
            print('Native window, UI and render targets:',actual,flush=True)
        viewer.window.cursor_mode=__import__('slangpy').CursorMode.disabled
        viewer.release_mouse()
        print('Captured fly cursor released successfully',flush=True)
    finally:viewer.close()


if __name__=='__main__':main()
