import unittest
import tkinter as tk
import time

class TestDesktopUI(unittest.TestCase):
    def test_ui_initialization_and_close(self):
        """
        Test that a simple Tkinter UI can be created and closed without error.
        """
        root = tk.Tk()
        # Schedule the window to close after 1 second.
        root.after(1000, root.destroy)
        start_time = time.time()
        root.mainloop()
        elapsed_time = time.time() - start_time
        # The UI should close within 2 seconds.
        self.assertLess(elapsed_time, 2, "UI did not close as expected.")

if __name__ == "__main__":
    unittest.main()
