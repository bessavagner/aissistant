from genaikit.settings import package
from genaikit.settings import HEADER
import tkinter as tk

from app import MyApp

class Window(tk.Tk):
    def __init__(self, app: MyApp):
        super().__init__()

        # Set the package of the app
        self.title(package['name'])

        # Set the size of the app window
        self.geometry("400x300")

        # Create the menu bar
        menubar = tk.Menu(self)

        # Create the "File" menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="About", command=self.show_about)

        # Add the menus to the menu bar
        menubar.add_cascade(label="File", menu=file_menu)

        # Configure the menu bar
        self.config(menu=menubar)

        # Create a frame for the labels
        label_frame = tk.Frame(self)
        label_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Create a label for the app's name
        name_label = tk.Label(label_frame, text=package['name'])
        name_label.pack(side=tk.LEFT, padx=10, pady=10)

        # Create a label for the app's version
        version_label = tk.Label(label_frame, text=f"v{package['version']}")
        version_label.pack(side=tk.RIGHT, padx=10, pady=10)

        # Create a frame for the text area and the button
        text_frame = tk.Frame(self)
        text_frame.pack(fill=tk.BOTH)

        # Create a text area
        self.textarea = tk.Text(text_frame, width=20, height=50)
        self.textarea.pack(side=tk.LEFT, padx=20, fill=tk.BOTH)

        # Create a button
        run_button = tk.Button(text_frame, text="Run", command=self.run)
        run_button.pack(side=tk.RIGHT, padx=10, pady=10)

    def show_about(self):
        about_window = tk.Toplevel(self)
        about_window.title("About")
        about_label = tk.Label(about_window, text=HEADER)
        about_label.pack(padx=20, pady=20)

    def run(self):
        # Add some text to the text area
        self.textarea.insert(tk.END, app.run())

if __name__ == "__main__":
    app = MyApp()
    window = Window(app)
    window.mainloop()
